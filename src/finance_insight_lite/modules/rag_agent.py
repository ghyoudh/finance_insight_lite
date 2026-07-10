import os
import re
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field, SecretStr
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from typing import List, Dict, Any, Literal


# ============================================================================
# Structured Output Schema
# ============================================================================

class DocumentGrade(BaseModel):
    """تقييم مستند واحد ضمن الدفعة"""
    doc_index: int = Field(description="رقم المستند كما ورد في القائمة، يبدأ من 1")
    relevance: Literal["Highly_Relevant", "Moderately_Relevant", "Irrelevant"] = Field(
        description="التصنيف - يجب أن يكون واحداً من هذه القيم الثلاث بالضبط، بدون أي نص إضافي"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="مستوى الثقة بالتصنيف من 0 إلى 1"
    )
    reason_brief: str = Field(
        max_length=120,
        description="سبب مختصر جداً (أقل من 15 كلمة) - يمكن أن يكون بالعربي أو الإنجليزي حسب لغة السؤال"
    )


class BatchGradeResponse(BaseModel):
    """الاستجابة الكاملة لتقييم دفعة المستندات"""
    grades: List[DocumentGrade] = Field(
        description="قائمة تحتوي على تقييم لكل مستند، بنفس عدد المستندات المُدخلة بالضبط"
    )


# ============================================================================
# 0. ADAPTIVE RETRIEVAL DEPTH — Corpus-Aware Adaptive-k
# ============================================================================
#
# المشكلة اللي يحلها هذا الجزء (fixed-k retrieval limitation):
# k ثابت (مثلاً 5) يسبب مشكلتين متعاكستين:
#   - ملف/قاعدة بيانات كبيرة: 5 مستندات قد تكون نسبة ضئيلة من المحتوى الفعلي
#     ذو الصلة -> فقدان معلومات (information loss / low recall)
#   - ملف/قاعدة بيانات صغيرة: النظام "يجبر" نفسه يرجع مستندات ضعيفة الصلة
#     لأنه ملتزم برقم ثابت -> نويز (noise / low precision)
#
# هذا الموضوع خط بحثي نشط بالأدبيات (Adaptive Retrieval):
#   - Jeong et al., 2024 (Adaptive-RAG): تكييف استراتيجية الاسترجاع حسب
#     تعقيد السؤال عبر classifier.
#   - Sun et al., 2025 (DynamicRAG): تحديد k ديناميكياً عبر reranker
#     مدرّب بالـ reinforcement learning.
#   - Taguchi et al., 2025 (Adaptive-k): اختيار عدد الفقرات المسترجعة
#     ديناميكياً بناءً على توزيع similarity scores.
#
# التصميم هنا (CA²-CG: Corpus-Aware Adaptive-k + Corrective Grading)
# يدمج فكرتين معاً بدل الاعتماد على واحدة فقط:
#   1) k_max يتحدد نسبياً لحجم الـ corpus (مش رقم ثابت بكل الحالات)
#   2) نقطة القطع الفعلية تُكتشف من توزيع الـ scores نفسه (elbow detection)
#      بدل ما نعتمد على threshold ثابت (0.6 مثلاً) قد يفشل مع تباين
#      توزيعات الدرجات بين الأسئلة المختلفة (خصوصاً بالنصوص العربية/
#      المختلطة، حيث توزيع similarity غير مستقر أحياناً)
#   3) الطبقة الأخيرة تبقى CRAG (LLM-based corrective grading, fail-closed)
#      كخط دفاع نهائي حقيقي لا يعتمد على أي score، بل على فهم المحتوى.
#
# النتيجة: نوسّع نافذة الاسترجاع الأولية بذكاء (fetch أوسع بدل k=5 دائماً)
# ونخلي الفلترة الحقيقية تتم على مرحلتين: إحصائية (elbow) ثم دلالية (LLM).
# ============================================================================

class AdaptiveRetrievalDepth:
    """
    يحسب حجم نافذة الاسترجاع (k) بشكل ديناميكي بناءً على:
      - حجم قاعدة البيانات (corpus size)
      - توزيع درجات التشابه (similarity score distribution) للسؤال الحالي
    """

    def __init__(
        self,
        k_min: int = 3,
        k_upper_bound: int = 20,
        corpus_divisor: int = 15,
        elbow_min_docs: int = 2,
    ):
        """
        Args:
            k_min: أقل عدد مستندات نجلبه دائماً كحد أدنى مطلق
            k_upper_bound: أقصى عدد مستندات نسمح فيه (سقف أمان للتكلفة/الزمن)
            corpus_divisor: كل ما زاد هذا الرقم، قل تأثير حجم الـ corpus على k
                            (قيمة تجريبية تُضبط حسب طبيعة البيانات)
            elbow_min_docs: أقل عدد مستندات تضمنه بعد اكتشاف نقطة الانكسار
        """
        self.k_min = k_min
        self.k_upper_bound = k_upper_bound
        self.corpus_divisor = corpus_divisor
        self.elbow_min_docs = elbow_min_docs

    def estimate_corpus_size(self, vector_db) -> Optional[int]:
        """
        محاولة استخراج حجم قاعدة البيانات المتجهية بشكل عام (best-effort)،
        بما يتوافق مع أكثر من backend شائع (Chroma, FAISS, وغيرها عبر LangChain).
        يرجع None لو ما قدرنا نحدد الحجم (fallback آمن على k_min..k_upper_bound الافتراضي).
        """
        # Chroma
        try:
            return vector_db._collection.count()
        except Exception:
            pass

        # FAISS (LangChain wrapper)
        try:
            return vector_db.index.ntotal
        except Exception:
            pass

        # واجهة عامة لو أضاف المطور دالة __len__ أو similar
        try:
            return len(vector_db)
        except Exception:
            pass

        return None

    def compute_k_max(self, vector_db) -> int:
        """يحسب أقصى عدد مستندات نجلبها من الاسترجاع الأولي (قبل القطع الإحصائي)"""
        corpus_size = self.estimate_corpus_size(vector_db)

        if corpus_size is None or corpus_size <= 0:
            # ما قدرنا نحدد الحجم -> نستخدم قيمة متوسطة آمنة بدل k=5 الثابت
            return max(self.k_min, min(10, self.k_upper_bound))

        k_max = corpus_size // self.corpus_divisor
        k_max = max(k_max, self.k_min)
        k_max = min(k_max, self.k_upper_bound)
        # حماية إضافية: لا تطلب أكثر مما يوجد فعلياً بالقاعدة
        k_max = min(k_max, corpus_size)
        return int(k_max)

    def detect_elbow(self, scores: List[float]) -> int:
        """
        يحدد نقطة الانكسار (elbow) في توزيع similarity scores مرتبة تنازلياً،
        عبر أكبر هبوط (gap) مطلق بين درجتين متتاليتين.

        الحدس وراءها: المستندات "الحقيقية ذات الصلة" عادة تشكل مجموعة
        متقاربة الدرجات بأعلى القائمة، ثم يحصل هبوط واضح قبل بقية
        المستندات الهامشية. هذا الهبوط هو خط القطع الطبيعي.
        """
        n = len(scores)
        if n == 0:
            return 0
        if n <= self.elbow_min_docs:
            return n

        scores_arr = np.array(scores, dtype=float)
        diffs = np.abs(np.diff(scores_arr))

        if len(diffs) == 0:
            return n

        elbow = int(np.argmax(diffs)) + 2  # تحويل index الفجوة إلى عدد عناصر مُختارة
        elbow = max(elbow, self.elbow_min_docs)
        elbow = min(elbow, n)
        return elbow


# ============================================================================
# 1. OPTIMIZED CRAG - Fast Retrieval with Batch Grading
# ============================================================================

class CRAGRetriever:
    """
    Corrective RAG مع تقييم دفعي عبر Structured Output (لا يوجد Regex)
    مدمج الآن مع AdaptiveRetrievalDepth (CA²-CG) بدل k الثابت.
    """

    # حد أدنى للثقة - أي تقييم بثقة أقل من هذا يُعامل كـ Irrelevant تلقائياً
    # (طبقة أمان إضافية: حتى لو صنّف النموذج المستند Relevant لكن بثقة ضعيفة،
    # نفضّل نحجبه بدل ما نجازف بعرض بيانات مالية غير دقيقة)
    MIN_CONFIDENCE_THRESHOLD = 0.6

    def __init__(self, vector_db, llm, adaptive_depth: Optional[AdaptiveRetrievalDepth] = None):
        self.vector_db = vector_db
        self.llm = llm
        self.adaptive_depth = adaptive_depth or AdaptiveRetrievalDepth()

        self.structured_llm = self.llm.with_structured_output(BatchGradeResponse)

        self.batch_grader_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Strategic Financial Analyst grading document relevance.

IMPORTANT - LANGUAGE HANDLING:
- The question and documents may be in Arabic, English, or mixed.
- Understand and reason about the content in whichever language it appears.
- However, your OUTPUT is always structured via the provided schema (tool call).
  The `relevance` field must ALWAYS be one of the three fixed English labels
  exactly as given: Highly_Relevant, Moderately_Relevant, Irrelevant.
  Never translate, paraphrase, or invent new labels for this field.
- `reason_brief` may be written in the same language as the question (Arabic
  question -> Arabic reason is fine), but keep it under 15 words.

CRITICAL RULES:
- Do NOT write any preamble, explanation, or free text outside the schema.
- Return exactly one grade per document, in the same order given.
- If you are not reasonably confident a document is relevant, classify it as
  Irrelevant rather than guessing Moderately_Relevant. When uncertain, prefer
  the stricter (less relevant) label.

**Core Analytical Framework (for your reasoning, not for output format):**
1. Historical Performance Analysis - trends in revenue, net income, margins
2. Predictive Signal Detection - links between strategic decisions and outcomes
3. Strategic Context Evaluation - business logic behind financial changes

**Document Relevance Criteria:**
- Highly_Relevant: contains quantitative data or context directly answering the question
- Moderately_Relevant: partial/supporting context only
- Irrelevant: no real financial substance connected to the question"""),
            ("human", "Question: {question}\n\nDocuments:\n{document}\n\nGrade each document now via the schema.")
        ])

    def batch_grade_documents(self, question: str, documents: List[Any]) -> List[bool]:
        """
        تقييم دفعي بدون Regex - يعتمد كلياً على Structured Output
        """
        if not documents:
            return []

        docs_text = "\n\n".join([
            f"Document {i + 1} [Page {doc.metadata.get('page')}]:\n{doc.page_content[:1500]}"
            for i, doc in enumerate(documents)
        ])

        try:
            response: BatchGradeResponse = self.structured_llm.invoke(
                self.batch_grader_prompt.format(
                    question=question,
                    document=docs_text
                )
            )

            # نبني خريطة doc_index -> grade لضمان الترتيب الصحيح
            # حتى لو النموذج رجّع الترتيب بشكل مختلف
            grade_map = {g.doc_index: g for g in response.grades}

            relevance_results = []
            for i in range(1, len(documents) + 1):
                grade = grade_map.get(i)

                if grade is None:
                    # النموذج ما رجّع تقييم لهذا المستند -> فشل آمن: نحجبه
                    print(f"⚠️ لا يوجد تقييم للمستند {i}, سيُحجب احترازياً")
                    relevance_results.append(False)
                    continue

                is_relevant = (
                    grade.relevance in ("Highly_Relevant", "Moderately_Relevant")
                    and grade.confidence >= self.MIN_CONFIDENCE_THRESHOLD
                )
                relevance_results.append(is_relevant)

            return relevance_results

        except Exception as e:
            # ✅ التصحيح الأهم: Fail-Closed وليس Fail-Open
            # القديم: return [True] * len(documents)  <- يمرر كل شيء وقت الخطأ (خطر)
            # الجديد: نحجب الكل وقت الخطأ، ونعتمد على fallback top-2 الموجود
            #         في get_relevant_documents لتفادي شاشة فاضية بالكامل
            print(f"❌ خطأ في التقييم الدفعي: {e} — سيتم حجب كل المستندات احترازياً")
            return [False] * len(documents)

    def _retrieve_with_scores(self, question: str, k: int):
        """
        يحاول جلب المستندات مع similarity scores (لازم للـ elbow detection).
        يدعم أكثر من واجهة شائعة عبر LangChain، ويسقط (fallback) للاسترجاع
        العادي بدون scores لو الـ vector store لا يدعمها.

        Returns:
            (documents: List, scores: Optional[List[float]])
        """
        # الواجهة الأكثر شيوعاً (Chroma, FAISS, ...)
        try:
            pairs = self.vector_db.similarity_search_with_relevance_scores(question, k=k)
            docs = [p[0] for p in pairs]
            scores = [float(p[1]) for p in pairs]
            return docs, scores
        except Exception:
            pass

        try:
            pairs = self.vector_db.similarity_search_with_score(question, k=k)
            docs = [p[0] for p in pairs]
            # بعض الـ backends (مثل Chroma الافتراضي) ترجع "distance" لا "similarity"
            # (أصغر = أقرب)، فنحولها لصيغة "كل ما زادت = أفضل" لضمان اتساق منطق
            # الـ elbow (اللي يفترض ترتيب تنازلي حسب الجودة)
            raw_scores = [float(p[1]) for p in pairs]
            if len(raw_scores) >= 2 and raw_scores[0] > raw_scores[-1]:
                # يبدو أنها فعلاً similarity تنازلية بالفعل، نتركها كما هي
                scores = raw_scores
            else:
                # يبدو أنها distance تصاعدية -> نعكسها لتصبح "أعلى = أفضل"
                max_val = max(raw_scores) if raw_scores else 1.0
                scores = [max_val - s for s in raw_scores]
            return docs, scores
        except Exception:
            pass

        # لا يوجد دعم لـ scores بهذا الـ vector store -> استرجاع عادي بدون elbow
        docs = self.vector_db.similarity_search(question, k=k)
        return docs, None

    def get_relevant_documents(self, question: str, k: Optional[int] = None) -> List[Dict]:
        """
        استرجاع متكيّف (CA²-CG) مع تقييم دفعي آمن:

        المرحلة 1 (إحصائية): نجلب نافذة واسعة (k_max) مبنية على حجم الـ corpus،
                              ثم نقصّها عند نقطة الانكسار (elbow) بتوزيع الـ scores.
        المرحلة 2 (دلالية):   الـ shortlist الناتج يمر على CRAG (LLM grading)
                              كخط دفاع أخير لا يعتمد على أي score، فقط على الفهم
                              الفعلي لمحتوى المستند مقابل السؤال.

        Args:
            question: السؤال المالي
            k: لو تم تمريره صراحة، يُستخدم كسقف أعلى (k_max) بدل الحساب التلقائي
               (مفيد للتوافق الخلفي backward compatibility)

        Returns:
            قائمة المستندات ذات الصلة
        """
        k_max = k if k is not None else self.adaptive_depth.compute_k_max(self.vector_db)
        print(f"🔍 نافذة الاسترجاع الأولية (k_max): {k_max}")

        candidate_docs, scores = self._retrieve_with_scores(question, k=k_max)

        if not candidate_docs:
            return []

        if scores is not None and len(scores) == len(candidate_docs):
            cutoff = self.adaptive_depth.detect_elbow(scores)
            print(f"📉 نقطة الانكسار (elbow): أخذ {cutoff} من أصل {len(candidate_docs)} حسب توزيع الدرجات")
            initial_docs = candidate_docs[:cutoff]
        else:
            # لا يوجد scores متاحة -> نمرر الكل لـ CRAG ونعتمد عليه بالكامل
            print("ℹ️ لا يدعم الـ vector store استخراج scores، الاعتماد الكامل على CRAG للفلترة")
            initial_docs = candidate_docs

        relevance_flags = self.batch_grade_documents(question, initial_docs)

        relevant_results = [
            {"document": doc, "relevant": True}
            for doc, is_relevant in zip(initial_docs, relevance_flags)
            if is_relevant
        ]

        print(f"📊 عدد المستندات ذات الصلة: {len(relevant_results)}/{len(initial_docs)}")

        # هذا fallback مختلف عن fail-open: هنا فعلاً ما لقينا شيء relevant
        # (بعد تقييم حقيقي، مو بسبب خطأ تقني) فنرجع top-2 كحل احتياطي
        # حتى ما تطلع للمستخدم صفحة فاضية بالكامل
        if not relevant_results:
            print("⚠️ لم يُعثر على مستندات ذات صلة، استخدام أفضل 2 كحل احتياطي")
            return [{"document": d, "relevant": True} for d in initial_docs[:2]]

        return relevant_results

# ============================================================================
# 2. Self-RAG Verification + Iterative Self-Refinement
# ============================================================================
#
# التصميم هنا يطبّق نمط Self-Refine (Madaan et al., 2023 — "Self-Refine:
# Iterative Refinement with Self-Feedback"): توليد -> نقد -> تحسين -> تكرار
# حتى معيار توقف، مع الاحتفاظ بتاريخ المحاولات لتجنب تكرار نفس الخطأ.
#
# الفرق المتعمّد عن الورقة الأصلية: الورقة تعتمد على self-feedback (النموذج
# ينتقد نفسه بدون مرجع خارجي). هنا النقد "مؤسَّس" (grounded) على مقارنة
# فعلية بالمستندات المصدرية عبر SelfRAGVerifier — نقد يعتمد على أدلة
# خارجية للنموذج، لا رأيه بنفسه فقط. هذا أقوى مصداقية لتطبيق مالي.
# ============================================================================

class VerificationResult(BaseModel):
    """نتيجة تحقق مبنية على schema - بدون أي parsing عبر regex"""
    rating: int = Field(ge=0, le=10, description="Overall accuracy score from 0 to 10")
    passed: bool = Field(description="True only if numbers are accurate and fully supported by sources")
    missing_refs: List[str] = Field(
        default_factory=list,
        description="Specific page numbers or facts referenced in the answer but not found in sources"
    )
    critical_notes: str = Field(
        max_length=300,
        description="One or two sentences, in English, describing the single most important issue "
                    "(or 'No issues found' if the answer is well supported)"
    )


class SelfRAGVerifier:
    """
    التحقق من الإجابة مقابل المصادر عبر Structured Output.
    يُستخدم كخطوة تحقق مستقلة، وأيضاً كمصدر الـ feedback لحلقة Self-Refine.
    """

    def __init__(self, llm):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(VerificationResult)

        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a meticulous financial fact-checker.

Compare the 'Answer' against the 'Source Documents' and grade it strictly via
the provided schema. Do not write any text outside the schema.

Grading rules:
- rating >= 7 requires every number in the Answer to be traceable to the Sources.
- If any figure cannot be verified against the Sources, rating must be below 7
  and passed must be false.
- missing_refs must list concrete items (e.g. "Q3 net margin figure", "page 12
  revenue breakdown") — not vague statements.
- critical_notes must always be written in English, regardless of the
  question's language, since it may be shown directly to the user."""),
            ("human", """Question: {question}

Answer: {answer}

Sources: {sources}

Grade this answer now via the schema.""")
        ])

    def verify_answer(self, question: str, answer: str, sources: List[str]) -> Dict[str, Any]:
        """التحقق من إجابة واحدة مقابل المصادر"""
        try:
            result: VerificationResult = self.structured_llm.invoke(
                self.verification_prompt.format(
                    question=question,
                    answer=answer,
                    sources="\n\n".join(sources[:5])  # Limit to first 5 sources for speed
                )
            )
            return {
                "rating": result.rating,
                "passed": result.passed,
                "missing_refs": result.missing_refs,
                "notes": result.critical_notes,
            }
        except Exception as e:
            # Fail-closed (متسق مع فلسفة CRAG): خطأ تقني بالتحقق لا يعني
            # "الإجابة ممتازة" -> نرجّع تقييم متوسط-منخفض بدل الثقة الكاملة،
            # عشان لا نعرض للمستخدم "High confidence" على إجابة ما تحققنا منها فعلياً
            print(f"⚠️ Verification error: {e}")
            return {
                "rating": 5,
                "passed": False,
                "missing_refs": [],
                "notes": "Automated verification was unavailable for this response; "
                         "treat the figures above with extra caution.",
            }


class SelfRefiningAnswerEngine:
    """
    ينسّق حلقة Self-Refine كاملة: توليد -> تحقق -> (لو فشل) تحسين موجّه
    بالنقد -> تحقق مجدداً -> يتكرر حتى معيار توقف أو سقف محاولات.

    يرجع دائماً أفضل محاولة رأيناها (best-so-far)، مو آخر محاولة بشكل أعمى،
    ويكون صريحاً مع المستخدم لو ما وصلنا لثقة عالية بعد كل المحاولات.
    """

    def __init__(self, llm, max_refinement_attempts: int = 2, pass_threshold: int = 7):
        """
        Args:
            llm: نموذج الـ LLM المستخدم للتوليد والتحسين
            max_refinement_attempts: عدد محاولات التحسين القصوى بعد المحاولة
                الأولى (2 يعني 3 توليدات بالمجموع كحد أقصى) - موازنة بين
                الجودة وتكلفة/زمن الاستجابة، متسقة مع الملاحظة التجريبية
                بأدبيات Self-Refine بأن معظم التحسن يحصل بأول تكرارين.
            pass_threshold: أقل rating يُعتبر "مقبول" ونوقف عنده الحلقة
        """
        self.llm = llm
        self.verifier = SelfRAGVerifier(llm)
        self.max_refinement_attempts = max_refinement_attempts
        self.pass_threshold = pass_threshold

        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Conversational Strategic Financial Advisor.

### LANGUAGE:
- Respond in the same language the user asked the question in (Arabic
  question -> Arabic answer, English question -> English answer). If the
  question is mixed, prefer clarity over strict language matching.

### CONVERSATIONAL LOGIC:
- Use the provided 'Chat History' to understand the context of the current question.
- If the user asks a follow-up (e.g., "Why?"), refer to the previous data extracted.

### INSTRUCTIONS:
1. **Extraction (JSON)**: Provide a clean JSON array only if new data points are found.
2. **Analysis**: Provide a brief, professional response regarding the data.
3. **Recommendations**: Offer 2 actionable suggestions.
4. **Currency**: Always include the currency (e.g., SAR, USD).

### OUTPUT FORMAT:
[JSON Data if applicable]

[Your conversational answer]

#### Suggestions
- [Suggestion 1]
- [Suggestion 2]"""),
            ("human", "Chat History: {chat_history}\n\nQuestion: {query}\n\nContext: {context}")
        ])

        self.refine_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Conversational Strategic Financial Advisor revising a
previous answer that failed an accuracy check.

### LANGUAGE:
- Keep the same language as the previous answer (match the user's question
  language: Arabic question -> Arabic answer, English question -> English answer).

### REVISION RULES:
1. You will be given your PREVIOUS ANSWER and a REVIEWER CRITIQUE of it.
2. Fix exactly the issues named in the critique — do not introduce new,
   unrelated changes.
3. Base every number strictly on the provided Context. If the critique
   flags a figure that is genuinely not present in the Context, do not
   invent it — state plainly that it is not available in the reviewed
   documents instead of guessing.
4. Keep the same output format: brief analysis, then a "#### Suggestions"
   section with exactly 2 actionable, professional suggestions.
5. Do not restate the critique itself to the user — just produce the
   corrected answer."""),
            ("human", """Question: {query}

Context: {context}

PREVIOUS ANSWER:
{previous_answer}

REVIEWER CRITIQUE:
- Rating: {rating}/10
- Missing/unsupported items: {missing_refs}
- Notes: {notes}

Provide the corrected answer now.""")
        ])

    def _generate_initial(self, query: str, context: str, chat_history: list) -> str:
        response = self.llm.invoke(
            self.answer_prompt.format_messages(query=query, context=context, chat_history=chat_history)
        )
        return response.content.strip()

    def _refine(self, query: str, context: str, previous_answer: str, verification: Dict[str, Any]) -> str:
        missing_refs_text = ", ".join(verification.get("missing_refs") or []) or "None specified"
        response = self.llm.invoke(
            self.refine_prompt.format_messages(
                query=query,
                context=context,
                previous_answer=previous_answer,
                rating=verification.get("rating", 0),
                missing_refs=missing_refs_text,
                notes=verification.get("notes", ""),
            )
        )
        return response.content.strip()

    def run(self, query: str, context: str, chat_history: list, source_texts: List[str]) -> Dict[str, Any]:
        """
        ينفذ حلقة التوليد/التحقق/التحسين الكاملة.

        Returns:
            dict فيه: answer, verification, attempts_made, self_refine_converged
        """
        try:
            answer = self._generate_initial(query, context, chat_history)
        except Exception as e:
            print(f"⚠️ Answer generation error: {e}")
            return {
                "answer": "We were unable to generate a response at this time. Please try again shortly.",
                "verification": {"rating": 0, "passed": False, "missing_refs": [], "notes": "Generation failed."},
                "attempts_made": 0,
                "self_refine_converged": False,
            }

        attempts = []  # كل محاولة: {"answer": ..., "verification": {...}}
        total_rounds = self.max_refinement_attempts + 1  # المحاولة الأولى + محاولات التحسين

        for round_idx in range(total_rounds):
            verification = self.verifier.verify_answer(query, answer, source_texts)
            attempts.append({"answer": answer, "verification": verification})

            if verification["passed"]:
                break

            is_last_round = (round_idx == total_rounds - 1)
            if is_last_round:
                break

            try:
                answer = self._refine(query, context, answer, verification)
            except Exception as e:
                print(f"⚠️ Refinement error on attempt {round_idx + 1}: {e}")
                break  # نوقف الحلقة ونعتمد على best-so-far بدل ما نكرر خطأ

        best_attempt = max(attempts, key=lambda a: a["verification"]["rating"])
        converged = attempts[-1]["verification"]["passed"]

        return {
            "answer": best_attempt["answer"],
            "verification": best_attempt["verification"],
            "attempts_made": len(attempts),
            "self_refine_converged": converged,
        }


# ============================================================================
# 3. VISUALIZATION TOOL - Financial Data Visualization
# ============================================================================

class FinancialDataExtractor:
    """Extract financial data from documents for visualization"""

    def __init__(self, vector_db, llm):
        self.vector_db = vector_db
        self.llm = llm  # Use LLM to help extract structured data

    def extract_data_from_query(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Extract numerical data using LLM-assisted parsing

        Args:
            query: What data to extract
            k: Number of documents to search

        Returns:
            DataFrame with extracted data
        """
        # Retrieve more documents for better data coverage
        docs = self.vector_db.similarity_search(query, k=k)

        if not docs:
            return pd.DataFrame()

        # Combine all doc content
        combined_text = "\n\n".join([
            f"[Page {doc.metadata.get('page')} | Sheet: {doc.metadata.get('sheet_name', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        ])

        # Use LLM to extract structured data
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract structured financial data. 
            RULES: 
            1. ONLY extract items relevant to the query. 
            2. Each object MUST have: "label", "value", "currency", "suggestion".
            3. "value" must be a CLEAN number. 
            4. If no new/relevant data found, return empty list [].
            5. STRICT: NO markdown, ONLY JSON array."""),
            ("human", "Query: {query}\n\nContext:\n{combined_text}\n\nJSON:")
        ])

        try:
            response = self.llm.invoke(extraction_prompt.format_messages(
                query=query,
                combined_text=combined_text
            ))
            raw = response.content.strip()

            # Clean up markdown code blocks
            raw = re.sub(r'```(?:json)?\s*', '', raw)
            raw = raw.replace('```', '').strip()

            # Capture only the JSON array part
            data = json.loads(raw)

            if isinstance(data, list):
                valid_data = []
                for item in data:
                    if not isinstance(item, dict): continue

                    # Convert keys to lowercase for consistency
                    item_clean = {str(k).lower().strip(): v for k, v in item.items()}

                    # Try to extract required fields
                    label = item_clean.get('label') or item_clean.get('name') or item_clean.get('description')
                    value = item_clean.get('value') or item_clean.get('amount')
                    currency = item_clean.get('currency', 'SAR')
                    suggestion = item_clean.get('suggestion', 'No specific advice')

                    if label and value is not None:
                        try:
                            # Clean the value to ensure it's numeric
                            clean_val = float(str(value).replace(',', '').replace('$', '').strip())
                            valid_data.append({
                                'label': str(label),
                                'value': clean_val,
                                'currency': str(currency),
                                'suggestion': str(suggestion)
                            })
                        except: continue

                if valid_data:
                    return pd.DataFrame(valid_data).drop_duplicates(subset=['label'])

        except Exception as e:
            print(f"❌ Detailed Debug: Error Type: {type(e).__name__}, Message: {str(e)}")

        # Fallback: Manual regex parsing
        print("📌 Falling back to regex extraction...")
        return pd.DataFrame(columns=['label', 'value', 'currency', 'suggestion'])

class ChartGenerator:
    """Generate Plotly charts with financial styling"""

    # Consistent color palette
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    @staticmethod
    def create_line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        fig = px.line(df, x=x, y=y, title=title, markers=True,
                      template="plotly_white", color_discrete_sequence=ChartGenerator.COLORS)
        fig.update_layout(
            hovermode='x unified', height=500,
            title_font_size=20, axis_title_font_size=14,
            xaxis_title=x.replace('_', ' ').title(),
            yaxis_title=y.replace('_', ' ').title()
        )
        return fig

    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        fig = px.bar(df, x=x, y=y, title=title, template="plotly_white",
                     color_discrete_sequence=ChartGenerator.COLORS)
        fig.update_layout(
            height=500, title_font_size=20,
            xaxis_title=x.replace('_', ' ').title(),
            yaxis_title=y.replace('_', ' ').title()
        )
        fig.update_traces(marker_color=ChartGenerator.COLORS[0])
        return fig

    @staticmethod
    def create_pie_chart(df: pd.DataFrame, names: str, values: str, title: str) -> go.Figure:
        fig = px.pie(df, names=names, values=values, title=title,
                     template="plotly_white", color_discrete_sequence=ChartGenerator.COLORS)
        fig.update_traces(textposition='inside', textinfo='percent')
        fig.update_layout(height=500, title_font_size=20)
        return fig

    @staticmethod
    def create_scatter_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        try:
            df = df.copy()
            # If x is not numeric, convert to numeric with labels
            if not pd.api.types.is_numeric_dtype(df[x]) and not pd.api.types.is_datetime64_any_dtype(df[x]):
                df["_x_index"] = range(len(df))
                x_plot = "_x_index"
                tickvals = df["_x_index"].tolist()
                ticktext = df[x].astype(str).tolist()
                # No trendline for categorical x
                fig = px.scatter(df, x=x_plot, y=y, title=title, template="plotly_white",
                                color_discrete_sequence=ChartGenerator.COLORS)
                fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            else:
                # Numeric/datetime x - can use trendline
                fig = px.scatter(df, x=x, y=y, title=title, template="plotly_white",
                                trendline="ols", color_discrete_sequence=ChartGenerator.COLORS)
            fig.update_layout(height=500, title_font_size=20)
            return fig
        except Exception as e:
            print(f"⚠️ Error creating scatter chart: {e}")
            raise

    @staticmethod
    def create_area_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        fig = px.area(df, x=x, y=y, title=title, template="plotly_white",
                      color_discrete_sequence=ChartGenerator.COLORS)
        fig.update_layout(height=500, title_font_size=20)
        return fig


# ============================================================================
# 4. OPTIMIZED Agentic RAG - Fast & Efficient
# ============================================================================

class FinancialRAGAgent:
    """Extract financial data from documents for visualization"""

    def __init__(self, vector_db):
        self.vector_db = vector_db

    def process_query(self, query: str , chat_history: list = None) -> dict:
        """
        Main entry point: runs the full pipeline from query to chart-ready data.
        """
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=SecretStr(api_key) if api_key else None,
            temperature=0,
        )

        # 1. Retrieve and grade documents using CA²-CG (Adaptive-k + CRAG)
        retriever = CRAGRetriever(self.vector_db, self.llm)
        relevant_results = retriever.get_relevant_documents(query)  # k يُحسب تلقائياً الآن

        relevant_docs = [r["document"] for r in relevant_results]

        if not relevant_docs:
            return {
                "answer": "No relevant documents found for your query.",
                "source_pages": [],
                "confidence": "Low",
                "verification": None,
                "relevant_docs_count": 0,
                "chart": None
            }

        # 2. Build context from relevant docs
        context = "\n\n".join([
            f"[Page {doc.metadata.get('page')}]\n{doc.page_content}"
            for doc in relevant_docs
        ])

        source_pages = list(set([
            str(doc.metadata.get('page')) for doc in relevant_docs
        ]))

        # 3 + 4. Generate -> Verify -> Refine loop (Self-Refine, grounded on sources)
        #
        # بدل توليد الإجابة مرة وحدة ثم مجرد قياس ثقتها، نستخدم
        # SelfRefiningAnswerEngine اللي يعيد صياغة الإجابة فعلياً لو التحقق
        # المبني على المصادر رفضها، بحد أقصى محاولتي تحسين، ويرجع دائماً
        # أفضل نتيجة رآها (best-so-far) لا آخر محاولة بشكل أعمى.
        if chat_history is None: chat_history = []

        refine_engine = SelfRefiningAnswerEngine(self.llm, max_refinement_attempts=2, pass_threshold=7)
        refine_result = refine_engine.run(
            query=query,
            context=context,
            chat_history=chat_history,
            source_texts=[doc.page_content for doc in relevant_docs],
        )

        answer = refine_result["answer"]
        verification = refine_result["verification"]

        confidence = "High" if verification.get("rating", 0) >= 8 else "Medium" if verification.get("rating", 0) >= 5 else "Low"

        # لو ما نجحت أي محاولة بالوصول لمعيار القبول، نكون صريحين مع
        # المستخدم بدل ما نعرض له رقم ثقة بدون سياق
        if not refine_result["self_refine_converged"]:
            answer += (
                "\n\n---\n"
                "**Note:** This response reflects our best available answer after multiple "
                "review passes, but some figures could not be fully verified against the "
                "source documents. Please treat it with appropriate caution."
            )

        # 5. Check if query is visualization-related and generate chart
        chart_data = None
        viz_keywords = ["chart", "visualiz", "plot", "graph", "draw", "pie", "bar", "line", "trend"]
        if any(kw in query.lower() for kw in viz_keywords):
            try:
                extractor = FinancialDataExtractor(self.vector_db, self.llm)
                df = extractor.extract_data_from_query(query)
                print(f"📊 DataFrame for visualization:\n{df.head()}")

                if not df.empty:
                    chart_type = self._suggest_chart_type(query, df)

                    # Generate the correct chart
                    if chart_type == "bar":
                        fig = ChartGenerator.create_bar_chart(df, x="label", y="value", title=query)
                    elif chart_type == "line":
                        fig = ChartGenerator.create_line_chart(df, x="label", y="value", title=query)
                    elif chart_type == "pie":
                        fig = ChartGenerator.create_pie_chart(df, names="label", values="value", title=query)
                    elif chart_type == "scatter":
                        fig = ChartGenerator.create_scatter_chart(df, x="label", y="value", title=query)
                    else:
                        fig = ChartGenerator.create_area_chart(df, x="label", y="value", title=query)

                    # Serialize the chart to JSON so it can be stored in chat history
                    chart_data = {
                        "success": True,
                        "chart": fig.to_json(),
                        "title": query,
                        "data_preview": df.to_dict(orient="records")
                    }
            except Exception as e:
                print(f"⚠️ Chart generation error: {e}")
                chart_data = {"success": False, "error": str(e)}

        # 6. Return everything the UI expects
        return {
            "answer": answer,
            "source_pages": source_pages,
            "confidence": confidence,
            "verification": verification,
            "relevant_docs_count": len(relevant_docs),
            "chart": chart_data,
            "self_refine_attempts": refine_result["attempts_made"],
            "self_refine_converged": refine_result["self_refine_converged"],
        }

    def _suggest_chart_type(self, query: str, df: pd.DataFrame) -> str:
        """
        Heuristic: pick chart type based on query keywords or data shape.
        """
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["trend", "over time", "quarterly", "yearly", "monthly"]):
            return "line"
        if any(kw in query_lower for kw in ["compare", "comparison", "breakdown", "share", "distribution"]):
            return "pie" if len(df) <= 6 else "bar"

        # Default: bar chart works for most financial comparisons
        return "bar"

    def _clean_dataframe(self, data: list) -> pd.DataFrame:
        """
        Strict cleaning of extracted data
        Removes any row with empty, null, or non-numeric values
        """
        cleaned = []

        for item in data:
            if not isinstance(item, dict):
                continue

            # Get label - must be a non-empty string
            label = item.get('label')
            if label is None or str(label).strip() == '':
                continue
            label = str(label).strip()

            # Get value - must be convertible to float
            value = item.get('value')
            if value is None or str(value).strip() == '':
                continue

            # Clean the value string: remove commas, spaces, currency symbols
            value_str = str(value).strip()
            value_str = value_str.replace(',', '')
            value_str = value_str.replace('﷼', '')
            value_str = value_str.replace('$', '')
            value_str = value_str.replace(' ', '')

            # Remove text units if attached (e.g., "28.0billion")
            value_str = re.sub(r'(billion|million|trillion|bn|mn|tn|مليار|مليون)', '', value_str, flags=re.IGNORECASE)
            value_str = value_str.strip()

            # Final check: must be a valid number
            if value_str == '':
                continue

            try:
                numeric_value = float(value_str)
                cleaned.append({'label': label, 'value': numeric_value})
            except (ValueError, TypeError):
                print(f"  ⚠️ Skipping invalid value: label='{label}', value='{value}'")
                continue

        if not cleaned:
            return pd.DataFrame()

        df = pd.DataFrame(cleaned)
        df = df.drop_duplicates(subset=['label'])

        return df
