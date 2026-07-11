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

class AdaptiveRetrievalDepth:
    """
    يحسب حجم نافذة الاسترجاع (k) بشكل ديناميكي بناءً على:
      - حجم قاعدة البيانات (corpus size)
      - توزيع درجات التشابه (similarity score distribution) للسؤال الحالي

    ملاحظة: هذا الكلاس عام وقابل لإعادة الاستخدام بإعدادات مختلفة حسب
    الاستهلاك. مثلاً:
      - الاسترجاع النصي (CRAGRetriever) يحتاج دقة أعلى -> corpus_divisor أكبر
        (نافذة أضيق) + k_upper_bound أقل.
      - استخراج بيانات الرسوم البيانية (FinancialDataExtractor) يحتاج تغطية
        أوسع (عشان يلقط كل فئات البيانات) -> corpus_divisor أصغر (نافذة أوسع)
        + k_upper_bound أعلى.
    """

    def __init__(
        self,
        k_min: int = 3,
        k_upper_bound: int = 20,
        corpus_divisor: int = 15,
        elbow_min_docs: int = 2,
    ):
        self.k_min = k_min
        self.k_upper_bound = k_upper_bound
        self.corpus_divisor = corpus_divisor
        self.elbow_min_docs = elbow_min_docs

    def estimate_corpus_size(self, vector_db) -> Optional[int]:
        try:
            return vector_db._collection.count()
        except Exception:
            pass

        try:
            return vector_db.index.ntotal
        except Exception:
            pass

        try:
            return len(vector_db)
        except Exception:
            pass

        return None

    def compute_k_max(self, vector_db) -> int:
        corpus_size = self.estimate_corpus_size(vector_db)

        if corpus_size is None or corpus_size <= 0:
            return max(self.k_min, min(10, self.k_upper_bound))

        k_max = corpus_size // self.corpus_divisor
        k_max = max(k_max, self.k_min)
        k_max = min(k_max, self.k_upper_bound)
        k_max = min(k_max, corpus_size)
        return int(k_max)

    def detect_elbow(self, scores: List[float]) -> int:
        n = len(scores)
        if n == 0:
            return 0
        if n <= self.elbow_min_docs:
            return n

        scores_arr = np.array(scores, dtype=float)
        diffs = np.abs(np.diff(scores_arr))

        if len(diffs) == 0:
            return n

        elbow = int(np.argmax(diffs)) + 2
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

    MIN_CONFIDENCE_THRESHOLD = 0.6

    def __init__(self, vector_db, llm, adaptive_depth: Optional[AdaptiveRetrievalDepth] = None):
        self.vector_db = vector_db
        self.llm = llm
        # إعدادات دقيقة (نافذة أضيق) مناسبة للإجابة النصية
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

            grade_map = {g.doc_index: g for g in response.grades}

            relevance_results = []
            for i in range(1, len(documents) + 1):
                grade = grade_map.get(i)

                if grade is None:
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
            print(f"❌ خطأ في التقييم الدفعي: {e} — سيتم حجب كل المستندات احترازياً")
            return [False] * len(documents)

    def _retrieve_with_scores(self, question: str, k: int):
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
            raw_scores = [float(p[1]) for p in pairs]
            if len(raw_scores) >= 2 and raw_scores[0] > raw_scores[-1]:
                scores = raw_scores
            else:
                max_val = max(raw_scores) if raw_scores else 1.0
                scores = [max_val - s for s in raw_scores]
            return docs, scores
        except Exception:
            pass

        docs = self.vector_db.similarity_search(question, k=k)
        return docs, None

    def get_relevant_documents(self, question: str, k: Optional[int] = None) -> List[Dict]:
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
            print("ℹ️ لا يدعم الـ vector store استخراج scores، الاعتماد الكامل على CRAG للفلترة")
            initial_docs = candidate_docs

        relevance_flags = self.batch_grade_documents(question, initial_docs)

        relevant_results = [
            {"document": doc, "relevant": True}
            for doc, is_relevant in zip(initial_docs, relevance_flags)
            if is_relevant
        ]

        print(f"📊 عدد المستندات ذات الصلة: {len(relevant_results)}/{len(initial_docs)}")

        if not relevant_results:
            print("⚠️ لم يُعثر على مستندات ذات صلة، استخدام أفضل 2 كحل احتياطي")
            return [{"document": d, "relevant": True} for d in initial_docs[:2]]

        return relevant_results

# ============================================================================
# 2. Self-RAG Verification + Iterative Self-Refinement
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
        try:
            result: VerificationResult = self.structured_llm.invoke(
                self.verification_prompt.format(
                    question=question,
                    answer=answer,
                    sources="\n\n".join(sources[:5])
                )
            )
            return {
                "rating": result.rating,
                "passed": result.passed,
                "missing_refs": result.missing_refs,
                "notes": result.critical_notes,
            }
        except Exception as e:
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
    """

    def __init__(self, llm, max_refinement_attempts: int = 2, pass_threshold: int = 7):
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
1. **Analysis**: Provide a brief, professional response regarding the data.
2. **Recommendations**: Offer 2 actionable suggestions.
3. **Currency**: Always include the currency (e.g., SAR, USD).
4. **No raw data**: Never output JSON, code blocks, key-value dumps, or any
   other raw/structured data format anywhere in your answer. Everything must
   be written as natural, conversational prose the end user can read directly.
5. **Never copy-paste from Context**: Do not restate, list, or dump the raw
   records/rows from the Context verbatim, not even reformatted with different
   spacing or punctuation. Extract only the specific numbers you need and
   weave them into your prose analysis (e.g. summarize totals, averages,
   trends — do not enumerate every single transaction/row).

### OUTPUT FORMAT:
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
   corrected answer.
6. Never output JSON, code blocks, or any raw/structured data format —
   plain conversational text only.
7. Never copy-paste raw records/rows from the Context verbatim. Summarize
   with your own words; only quote the specific figures needed."""),
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

    @staticmethod
    def _strip_json_artifacts(text: str) -> str:
        """
        شبكة أمان إضافية (defense-in-depth): حتى لو النموذج تجاهل تعليمات
        البرومبت ورجّع JSON (أو ما يشبهه) ضمن الإجابة — سواء بالبداية، أو
        داخل ```json code block، أو حتى بنص وسط أو نهاية الجواب — نحذفه هنا
        قبل ما يوصل للمستخدم النهائي.

        التحديث المهم: لا نعتمد فقط على json.loads الصارم للتحقق. كتل كثيرة
        تكون "منطقياً" تفريغ بيانات خام لكنها تفشل بالـ parsing الصارم بسبب
        علامات اقتباس ذكية (curly quotes) يولّدها النموذج أحياناً، أو فاصلة
        زايدة بالنهاية. لذلك نطبّع الاقتباسات أولاً، وإذا فشل json.loads
        نستخدم fallback: لو الكتلة المتزنة بالأقواس فيها نمط "key": value
        متكرر بشكل كثيف (3 مرات فأكثر)، نعتبرها تفريغ بيانات ونحذفها أيضاً.
        """
        if not text:
            return text

        cleaned = text

        # 1) حذف أي code block كامل (```json ... ``` أو ``` ... ```) أولاً
        cleaned = re.sub(r'```(?:json)?\s*[\s\S]*?```', '', cleaned, flags=re.IGNORECASE)

        # 2) تطبيع علامات الاقتباس الذكية (curly quotes) قبل أي فحص،
        #    لأنها سبب شائع لفشل json.loads رغم إن الكتلة منطقياً JSON صالح
        smart_quote_map = {
            '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
        }
        for bad, good in smart_quote_map.items():
            cleaned = cleaned.replace(bad, good)

        # 3) مسح النص بالكامل والبحث عن أي كتلة [...] أو {...} متزنة
        #    بالأقواس بأي موضع، وحذفها لو كانت فعلاً تفريغ بيانات
        n = len(cleaned)
        out_chars = []
        i = 0
        while i < n:
            ch = cleaned[i]
            if ch in ('[', '{'):
                closing = ']' if ch == '[' else '}'
                depth = 0
                j = i
                end_idx = None
                while j < n:
                    if cleaned[j] == ch:
                        depth += 1
                    elif cleaned[j] == closing:
                        depth -= 1
                        if depth == 0:
                            end_idx = j
                            break
                    j += 1

                if end_idx is not None:
                    candidate = cleaned[i:end_idx + 1]
                    is_data_dump = False

                    try:
                        json.loads(candidate)
                        # كتلة JSON صالحة فعلاً -> تفريغ بيانات، نحذفها
                        is_data_dump = True
                    except (json.JSONDecodeError, ValueError):
                        # فشل التحقق الصارم (فاصلة زايدة، سطر مقطوع...) —
                        # نتحقق بشكل تقريبي: هل فيها نمط "key": متكرر بكثافة؟
                        # (زي "Transaction ID": ...، "Total": ...، "Date": ...)
                        kv_pattern_count = len(
                            re.findall(r'"[^"\n]{1,60}"\s*:\s*', candidate)
                        )
                        if kv_pattern_count >= 3:
                            is_data_dump = True

                    if is_data_dump:
                        i = end_idx + 1
                        continue
                    # مو JSON ولا تفريغ بيانات (قوس ضمن جملة عادية مثلاً) —
                    # نعامله كنص طبيعي ونكمل

            out_chars.append(ch)
            i += 1

        cleaned = ''.join(out_chars)

        # تنظيف الأسطر الفارغة الزائدة اللي تخلّف من حذف الكتل
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        # لو الحذف خلّى النص فاضي بالكامل (حالة نادرة)، نرجع الأصل بدل ما
        # نعرض للمستخدم إجابة فاضية
        return cleaned if cleaned else text.strip()

    def _generate_initial(self, query: str, context: str, chat_history: list) -> str:
        response = self.llm.invoke(
            self.answer_prompt.format_messages(query=query, context=context, chat_history=chat_history)
        )
        return self._strip_json_artifacts(response.content.strip())

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
        return self._strip_json_artifacts(response.content.strip())

    def run(self, query: str, context: str, chat_history: list, source_texts: List[str]) -> Dict[str, Any]:
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

        attempts = []
        total_rounds = self.max_refinement_attempts + 1

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
                break

        best_attempt = max(attempts, key=lambda a: a["verification"]["rating"])
        converged = attempts[-1]["verification"]["passed"]

        return {
            "answer": self._strip_json_artifacts(best_attempt["answer"]),
            "verification": best_attempt["verification"],
            "attempts_made": len(attempts),
            "self_refine_converged": converged,
        }


# ============================================================================
# 3. VISUALIZATION TOOL - Financial Data Visualization
# ============================================================================

class FinancialDataExtractor:
    """
    Extract financial data from documents for visualization.

    التحديث: بدل k=5 الثابت، نستخدم الآن AdaptiveRetrievalDepth لحساب حجم
    نافذة الاسترجاع ديناميكياً حسب حجم الملف/القاعدة.

    نستخدم إعدادات "أوسع" افتراضياً (corpus_divisor أصغر و k_upper_bound
    أعلى) مقارنة بالإعدادات المستخدمة بالإجابة النصية (CRAGRetriever)، لأن
    هدف الرسم البياني هو التغطية (coverage) — نبي نلقط كل فئات البيانات
    المرتبطة بالسؤال (مثلاً كل بنود المصاريف) — وليس التضييق الشديد
    بالدقة اللي يحتاجه توليد الإجابة النصية.
    """

    def __init__(self, vector_db, llm, adaptive_depth: Optional[AdaptiveRetrievalDepth] = None):
        self.vector_db = vector_db
        self.llm = llm
        self.adaptive_depth = adaptive_depth or AdaptiveRetrievalDepth(
            k_min=4,
            k_upper_bound=25,
            corpus_divisor=10,  # نافذة أوسع من الإعداد الافتراضي (15) عشان تغطية أفضل
        )

    def extract_data_from_query(self, query: str, k: Optional[int] = None) -> pd.DataFrame:
        k_max = k if k is not None else self.adaptive_depth.compute_k_max(self.vector_db)
        print(f"📊 نافذة الاسترجاع لاستخراج بيانات الرسم البياني (k_max): {k_max}")

        docs = self.vector_db.similarity_search(query, k=k_max)

        if not docs:
            return pd.DataFrame()

        combined_text = "\n\n".join([
            f"[Page {doc.metadata.get('page')} | Sheet: {doc.metadata.get('sheet_name', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        ])

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

            raw = re.sub(r'```(?:json)?\s*', '', raw)
            raw = raw.replace('```', '').strip()

            data = json.loads(raw)

            if isinstance(data, list):
                valid_data = []
                for item in data:
                    if not isinstance(item, dict): continue

                    item_clean = {str(k).lower().strip(): v for k, v in item.items()}

                    label = item_clean.get('label') or item_clean.get('name') or item_clean.get('description')
                    value = item_clean.get('value') or item_clean.get('amount')
                    currency = item_clean.get('currency', 'SAR')
                    suggestion = item_clean.get('suggestion', 'No specific advice')

                    if label and value is not None:
                        try:
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

        print("📌 Falling back to regex extraction...")
        return pd.DataFrame(columns=['label', 'value', 'currency', 'suggestion'])

class ChartGenerator:
    """Generate Plotly charts with financial styling"""

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
            if not pd.api.types.is_numeric_dtype(df[x]) and not pd.api.types.is_datetime64_any_dtype(df[x]):
                df["_x_index"] = range(len(df))
                x_plot = "_x_index"
                tickvals = df["_x_index"].tolist()
                ticktext = df[x].astype(str).tolist()
                fig = px.scatter(df, x=x_plot, y=y, title=title, template="plotly_white",
                                color_discrete_sequence=ChartGenerator.COLORS)
                fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            else:
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
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=SecretStr(api_key) if api_key else None,
            temperature=0,
        )

        retriever = CRAGRetriever(self.vector_db, self.llm)
        relevant_results = retriever.get_relevant_documents(query)

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

        context = "\n\n".join([
            f"[Page {doc.metadata.get('page')}]\n{doc.page_content}"
            for doc in relevant_docs
        ])

        def _source_label(doc) -> Optional[str]:
            page = doc.metadata.get('page')
            if page is not None:
                return str(page)
            sheet = doc.metadata.get('sheet_name')
            if sheet:
                return f"Sheet: {sheet}"
            return None

        source_pages = sorted({
            label
            for label in (_source_label(doc) for doc in relevant_docs)
            if label is not None
        })

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

        if not refine_result["self_refine_converged"]:
            answer += (
                "\n\n---\n"
                "**Note:** This response reflects our best available answer after multiple "
                "review passes, but some figures could not be fully verified against the "
                "source documents. Please treat it with appropriate caution."
            )

        chart_data = None
        viz_keywords = ["chart", "visualiz", "plot", "graph", "draw", "pie", "bar", "line", "trend"]
        if any(kw in query.lower() for kw in viz_keywords):
            try:
                # نستخدم adaptive_depth مستقل وأوسع (تغطية) بدل k=5 الثابت،
                # ومنفصل عن إعدادات retriever النصي (اللي همه الدقة/التضييق)
                extractor = FinancialDataExtractor(self.vector_db, self.llm)
                df = extractor.extract_data_from_query(query)
                print(f"📊 DataFrame for visualization:\n{df.head()}")

                if not df.empty:
                    chart_type = self._suggest_chart_type(query, df)

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

                    chart_data = {
                        "success": True,
                        "chart": fig.to_json(),
                        "title": query,
                        "data_preview": df.to_dict(orient="records")
                    }
            except Exception as e:
                print(f"⚠️ Chart generation error: {e}")
                chart_data = {"success": False, "error": str(e)}

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
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["trend", "over time", "quarterly", "yearly", "monthly"]):
            return "line"
        if any(kw in query_lower for kw in ["compare", "comparison", "breakdown", "share", "distribution"]):
            return "pie" if len(df) <= 6 else "bar"

        return "bar"

    def _clean_dataframe(self, data: list) -> pd.DataFrame:
        cleaned = []

        for item in data:
            if not isinstance(item, dict):
                continue

            label = item.get('label')
            if label is None or str(label).strip() == '':
                continue
            label = str(label).strip()

            value = item.get('value')
            if value is None or str(value).strip() == '':
                continue

            value_str = str(value).strip()
            value_str = value_str.replace(',', '')
            value_str = value_str.replace('﷼', '')
            value_str = value_str.replace('$', '')
            value_str = value_str.replace(' ', '')

            value_str = re.sub(r'(billion|million|trillion|bn|mn|tn|مليار|مليون)', '', value_str, flags=re.IGNORECASE)
            value_str = value_str.strip()

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
