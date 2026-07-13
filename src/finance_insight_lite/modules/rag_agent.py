import os
import re
import time
import threading
from collections import deque
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

from .query_expansion import QueryExpander
from .hybrid_retriever import HybridRetriever


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

        if corpus_size <= 150:
            small_corpus_floor = min(self.k_upper_bound, max(8, int(corpus_size * 0.3)))
            k_max = max(k_max, small_corpus_floor)

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

        min_safe_docs = max(self.elbow_min_docs, min(5, n))
        elbow = max(elbow, min_safe_docs)

        elbow = min(elbow, n)
        return elbow


# ============================================================================
# 0.5 TPM RATE LIMITER — انتظار محسوب بدل الاصطدام العشوائي بسقف Groq
# ============================================================================

class TPMRateLimiter:


    def __init__(self, tpm_limit: int, safety_margin: float = 0.9, window_seconds: float = 60.0):
        self.tpm_limit = tpm_limit
        self.safety_threshold = tpm_limit * safety_margin
        self.window_seconds = window_seconds
        self._events = deque()  # كل عنصر: [timestamp, estimated_tokens]
        self._lock = threading.Lock()

    @staticmethod
    def estimate_tokens(text: str) -> int:
       
        return max(1, len(text) // 3)

    def _prune(self, now: float):
        while self._events and (now - self._events[0][0]) > self.window_seconds:
            self._events.popleft()

    def wait_if_needed(self, estimated_tokens: int, label: str = ""):
        with self._lock:
            now = time.time()
            self._prune(now)
            used = sum(t for _, t in self._events)

            if used + estimated_tokens > self.safety_threshold and self._events:
                oldest_ts = self._events[0][0]
                sleep_for = self.window_seconds - (now - oldest_ts) + 0.3
                if sleep_for > 0:
                    print(
                        f"⏳ TPM pacing [{label}]: انتظار محسوب {sleep_for:.2f}s "
                        f"(مستخدم تقريباً {used}/{self.tpm_limit} + طلب جديد ~{estimated_tokens})"
                    )
                    time.sleep(sleep_for)
                    now = time.time()
                    self._prune(now)

            self._events.append([now, estimated_tokens])


# ============================================================================
# 1. OPTIMIZED CRAG - Fast Retrieval with Batch Grading
# ============================================================================

class CRAGRetriever:

    MIN_CONFIDENCE_THRESHOLD = 0.45
    GRADING_SNIPPET_CHARS = 600

    def __init__(
        self,
        vector_db,
        llm,
        adaptive_depth: Optional[AdaptiveRetrievalDepth] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        query_expander: Optional[QueryExpander] = None,
    ):
        self.vector_db = vector_db
        self.llm = llm
        self.adaptive_depth = adaptive_depth or AdaptiveRetrievalDepth()

        self.hybrid_retriever = hybrid_retriever
        self.query_expander = query_expander

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

IMPORTANT - SEMANTIC MATCHING (not literal keyword matching):
- The user's question may be phrased very differently from the document's
  wording — it may use synonyms, paraphrasing, a different level of
  formality, or describe the same concept from another angle (e.g. "أداء
  الشركة المالي بآخر 3 شهور" vs a document that says "إيرادات الربع
  الثالث"). These are the SAME topic and should be graded as if the
  question used the document's own wording.
- Judge relevance based on whether the document's underlying financial
  meaning/topic answers the question — NOT on how many words overlap
  literally between the question and the document text.
- Do not lower your relevance classification or confidence merely because
  the question and document use different terminology for the same concept.

CRITICAL RULES:
- Do NOT write any preamble, explanation, or free text outside the schema.
- Return exactly one grade per document, in the same order given.
- Only classify a document as Irrelevant if its actual subject matter is
  unrelated to what is being asked — not because the wording differs. When
  the topic clearly matches but you are unsure about fine details, prefer
  Moderately_Relevant over Irrelevant.

**Core Analytical Framework (for your reasoning, not for output format):**
1. Historical Performance Analysis - trends in revenue, net income, margins
2. Predictive Signal Detection - links between strategic decisions and outcomes
3. Strategic Context Evaluation - business logic behind financial changes

**Document Relevance Criteria:**
- Highly_Relevant: contains quantitative data or context directly answering the question (regardless of exact wording overlap)
- Moderately_Relevant: partial/supporting context only, or same topic but less directly on-point
- Irrelevant: no real financial substance connected to the question's topic"""),
            ("human", "Question: {question}\n\nDocuments:\n{document}\n\nGrade each document now via the schema.")
        ])

    def batch_grade_documents(self, question: str, documents: List[Any]) -> List[bool]:
        if not documents:
            return []

        docs_text = "\n\n".join([
            f"Document {i + 1} [Page {doc.metadata.get('page')}]:\n{doc.page_content[:self.GRADING_SNIPPET_CHARS]}"
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

    def _retrieve_candidates(self, question: str, k_max: int):
        if self.hybrid_retriever is not None:
            if self.query_expander is not None:
                _t = time.time()
                queries = self.query_expander.expand(question)
                print(f"⏱️   ├─ Query Expansion: {time.time() - _t:.2f}s")
            else:
                queries = [question]

            _t = time.time()
            result = self.hybrid_retriever.retrieve_with_scores(queries, k_max=k_max)
            print(f"⏱️   ├─ Hybrid Search (BM25+vector, {len(queries)} صياغة): {time.time() - _t:.2f}s")
            return result

        return self._retrieve_with_scores(question, k=k_max)

    def get_relevant_documents(self, question: str, k: Optional[int] = None) -> List[Dict]:
        k_max = k if k is not None else self.adaptive_depth.compute_k_max(self.vector_db)
        print(f"🔍 نافذة الاسترجاع الأولية (k_max): {k_max}")

        candidate_docs, scores = self._retrieve_candidates(question, k_max=k_max)

        if not candidate_docs:
            return []

        if scores is not None and len(scores) == len(candidate_docs):
            cutoff = self.adaptive_depth.detect_elbow(scores)
            print(f"📉 نقطة الانكسار (elbow): أخذ {cutoff} من أصل {len(candidate_docs)} حسب توزيع الدرجات")
            initial_docs = candidate_docs[:cutoff]
        else:
            print("ℹ️ لا يدعم الـ vector store استخراج scores، الاعتماد الكامل على CRAG للفلترة")
            initial_docs = candidate_docs

        _t = time.time()
        relevance_flags = self.batch_grade_documents(question, initial_docs)
        print(f"⏱️   └─ CRAG Grading ({len(initial_docs)} مستند): {time.time() - _t:.2f}s")

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

class NumberAttribution(BaseModel):

    number_in_answer: str = Field(description="The exact number/figure as it appears in the Answer")
    row_label_in_source: str = Field(
        description="The exact row/item label this number is attached to in the Sources "
                    "(copy it as it appears in the source text). If the number cannot be "
                    "found in the Sources at all, write 'NOT_FOUND_IN_SOURCES'."
    )
    matches_question_intent: bool = Field(
        description="True only if row_label_in_source is actually what the Question is asking "
                    "about — not merely a similarly-worded neighboring row/category/period."
    )


class VerificationResult(BaseModel):
    """نتيجة تحقق مبنية على schema - بدون أي parsing عبر regex"""
    number_checks: List[NumberAttribution] = Field(
        default_factory=list,
        description="One entry per distinct number/figure mentioned in the Answer. Must be "
                    "filled BEFORE deciding rating/passed — your rating/passed decision should "
                    "follow logically from these checks, not the other way around."
    )
    rating: int = Field(ge=0, le=10, description="Overall accuracy score from 0 to 10")
    passed: bool = Field(description="True only if numbers are accurate and fully supported by sources")
    missing_refs: List[str] = Field(
        default_factory=list,
        description="Specific page numbers or facts referenced in the answer but not found in sources"
    )
    critical_notes: str = Field(
        default="No issues found.",
        max_length=600,
        description="One or two sentences, in English, describing the single most important issue "
                    "(or 'No issues found' if the answer is well supported)"
    )


class SelfRAGVerifier:

  
    VERIFICATION_SNIPPET_CHARS = 800
    MAX_SOURCES_FOR_VERIFICATION = 3

    def __init__(self, llm, rate_limiter: Optional["TPMRateLimiter"] = None):
        self.llm = llm
        self.rate_limiter = rate_limiter
        self.structured_llm = self.llm.with_structured_output(VerificationResult)

        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a meticulous financial fact-checker.

Compare the 'Answer' against the 'Source Documents' and grade it strictly via
the provided schema. Do not write any text outside the schema.

STEP 1 — number_checks (do this FIRST, before deciding rating/passed):
- List every distinct number/figure mentioned in the Answer.
- For each one, find its row/label in the Sources and copy that label
  verbatim into row_label_in_source.
- Judge matches_question_intent honestly: it is True only if that row/label
  is actually what the Question asked about. A similarly-worded neighboring
  row/category/period is NOT a match, even if the number itself is real and
  present somewhere in the Sources.

STEP 2 — rating/passed (derive these FROM step 1, don't decide independently):
- rating >= 7 requires every number_check to have matches_question_intent=true
  and every number to be traceable to the Sources.
- If ANY number_check has matches_question_intent=false, or any figure cannot
  be found in the Sources at all, rating must be below 7 and passed must be
  false — treat a mismatched-row number exactly like a fabricated number.
- missing_refs must list concrete items, and when the issue is a mismatch
  (real number, wrong row/label), say so explicitly (e.g. "figure X is
  attributed to the wrong row/label in the source table") — not vague
  statements.
- critical_notes must always be written in English, regardless of the
  question's language, since it may be shown directly to the user. Keep it
  under 40 words."""),
            ("human", """Question: {question}

Answer: {answer}

Sources: {sources}

Grade this answer now via the schema.""")
        ])

    def verify_answer(self, question: str, answer: str, sources: List[str]) -> Dict[str, Any]:
        try:
            trimmed_sources = "\n\n".join(
                s[:self.VERIFICATION_SNIPPET_CHARS]
                for s in sources[:self.MAX_SOURCES_FOR_VERIFICATION]
            )
            formatted_prompt = self.verification_prompt.format(
                question=question,
                answer=answer,
                sources=trimmed_sources
            )

            if self.rate_limiter is not None:
                self.rate_limiter.wait_if_needed(
                    TPMRateLimiter.estimate_tokens(formatted_prompt),
                    label="Verification"
                )

            result: VerificationResult = self.structured_llm.invoke(formatted_prompt)

            rating = result.rating
            passed = result.passed
            missing_refs = list(result.missing_refs)

            for check in result.number_checks:
                is_bad = (
                    not check.matches_question_intent
                    or check.row_label_in_source.strip().upper() == "NOT_FOUND_IN_SOURCES"
                )
                if is_bad:
                    passed = False
                    rating = min(rating, 4)
                    mismatch_note = (
                        f"{check.number_in_answer} attributed to '{check.row_label_in_source}' "
                        f"which does not match the question's intent"
                    )
                    if mismatch_note not in missing_refs:
                        missing_refs.append(mismatch_note)

            return {
                "rating": rating,
                "passed": passed,
                "missing_refs": missing_refs,
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

    # ========================================================================
   
    # ========================================================================

    def __init__(self, llm, verifier_llm=None, max_refinement_attempts: int = 1,
                 pass_threshold: int = 7, rate_limiter: Optional["TPMRateLimiter"] = None,
                 verifier_rate_limiter: Optional["TPMRateLimiter"] = None):
        self.llm = llm
        self.rate_limiter = rate_limiter
    
        self.verifier = SelfRAGVerifier(verifier_llm or llm, rate_limiter=verifier_rate_limiter)
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

### GROUNDING RULES (read carefully — this is the most important section):
- Base every answer on the retrieved Context only. Do not infer causal
  relationships unless the Context explicitly states them.
- Do NOT state that one metric causes, explains, drives, or correlates with
  another metric unless the Context explicitly says so in words. Two
  numbers appearing near each other, or both being "high"/"low" in the same
  period, is NOT evidence of a relationship between them — present them as
  separate, independent facts instead.
  - Wrong: "Strengths usage rose +80%, which explains the high ROI score."
  - Right: "Strengths usage rose +80%. Separately, the ROI score was 3.71.
    The report does not establish a direct relationship between these two
    metrics."
- If you are tempted to explain *why* something happened and the Context
  does not state the reason explicitly, say plainly that the evidence is
  insufficient to explain the cause — do not fill the gap with a
  plausible-sounding guess.
- Classify KPIs using standard business definitions, and base any
  explanation on what the KPI actually measures (its methodology) — not on
  assumptions about timing, causality, or unstated context.
- NEVER invent budgets, percentages, currency amounts (SAR/USD/etc.),
  timelines, or staffing/headcount numbers unless that exact figure already
  appears in the Context.

### RESPONSE STRUCTURE — decide based on what the question actually asks:

**Case A — Pure extraction/factual question** (asks for a specific number,
value, count, or fact — e.g. "What is X?", "How many...?", "كم عدد...؟",
"ما قيمة...؟"):
- Answer with ONLY the requested fact(s), stated in one or two clear
  sentences, supported by the number(s) from the Context.
- Do NOT add an "Analysis" section and do NOT add a "Suggestions" section.
  If the question doesn't ask for interpretation or recommendations, don't
  volunteer them.
- If the fact is genuinely not in the Context, say so plainly instead of
  guessing.

**Case B — Analysis / explanation / recommendation question** (asks "why",
"how", "what should we do", asks to compare, evaluate, or advise):
- Structure your answer in three clearly labeled parts (use the same
  language as the question for the labels — Arabic labels shown here,
  mirror in English as Facts / Analysis / Suggestions):

الحقائق:
[Only what is explicitly stated in the Context, with numbers. No
interpretation here.]

التحليل:
[Your interpretation of the facts above, clearly framed as interpretation
(e.g. "قد يشير هذا إلى..." / "This may suggest..."), never stated as if it
were a fact from the report. If evidence is insufficient to support an
interpretation the question is asking for, say so explicitly here instead
of guessing.]

الاقتراحات:
[Only include this part if the question actually asks for advice,
recommendations, or next steps. Each suggestion must be immediately
followed by the specific evidence from the Context it is based on, in this
form:
- التوصية: [qualitative, actionable suggestion — no invented numbers]
  الدليل: [the specific fact/number from the Context that motivates it]
Keep suggestions qualitative/directional (e.g. "consider increasing
investment in coaching quality" — not "invest 15,000 SAR in coaching"),
unless the Context itself already contains the number you're citing.]

### OTHER RULES:
- **Currency**: Always include the currency (e.g., SAR, USD) — but only
  when citing a figure that actually appears in the Context.
- **No raw data**: Never output JSON, code blocks, key-value dumps, or any
  other raw/structured data format anywhere in your answer. Everything must
  be written as natural, conversational prose the end user can read directly.
- **Never copy-paste from Context**: Do not restate, list, or dump the raw
  records/rows from the Context verbatim, not even reformatted with different
  spacing or punctuation. Extract only the specific numbers you need and
  weave them into your prose (e.g. summarize totals, averages, trends — do
  not enumerate every single transaction/row).
- **Row/label precision in tables**: The Context may contain tables with
  several rows whose labels are similarly worded (e.g. close variations of
  the same phrase describing different metrics, periods, or categories).
  Before using any number, double-check it is taken from the row/label that
  actually matches what the Question is asking about — not a neighboring row
  that merely looks similar. If the Question's target is ambiguous between
  two similarly-named rows, briefly note the ambiguity rather than guessing."""),
            ("human", "Chat History: {chat_history}\n\nQuestion: {query}\n\nContext: {context}")
        ])

        self.refine_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Conversational Strategic Financial Advisor revising a
previous answer that failed an accuracy check.

### LANGUAGE:
- Keep the same language as the previous answer (match the user's question
  language: Arabic question -> Arabic answer, English question -> English answer).

### GROUNDING RULES (apply these even while revising):
- Do NOT state that one metric causes, explains, drives, or correlates with
  another metric unless the Context explicitly says so in words. If the
  previous answer did this, remove the causal claim and present the two
  facts separately instead.
- Suggestions are professional opinion, not facts — never invent specific
  numbers, amounts, budgets, currency figures (SAR/USD/etc.), percentages,
  timelines, or staffing/headcount numbers that do not appear in the
  Context. Keep suggestions qualitative unless a number is already present
  in the Context.

### REVISION RULES:
1. You will be given your PREVIOUS ANSWER and a REVIEWER CRITIQUE of it.
2. Fix exactly the issues named in the critique — do not introduce new,
   unrelated changes.
3. Base every number strictly on the provided Context. If the critique
   flags a figure that is genuinely not present in the Context, do not
   invent it — state plainly that it is not available in the reviewed
   documents instead of guessing.
3b. If the critique flags a number as attributed to the wrong row/label
   (a real number pulled from a similarly-worded but different row/category/
   period than what the Question asked about), find and use the number from
   the CORRECT matching row/label in the Context instead — do not just drop
   the number or repeat the same mismatch.
4. Keep the same response structure as the original answer:
   - If the Question is a pure extraction/factual question, keep it to just
     the fact(s) — no "التحليل"/"Analysis" section, no "الاقتراحات"/
     "Suggestions" section.
   - If the Question asks for analysis/explanation/recommendations, keep
     the three labeled parts (الحقائق / التحليل / الاقتراحات, or
     Facts/Analysis/Suggestions in English) and make sure every suggestion
     is still immediately followed by its supporting evidence from the
     Context ("الدليل:" / "Evidence:").
   Do not add a Suggestions section that wasn't warranted by the Question,
   and do not remove one that the Question does call for.
5. Do not restate the critique itself to the user — just produce the
   corrected answer.
5b. Never describe your own edit process (e.g. do not write things like
   "removed the unsupported unit X" or "kept the verified figure Y"). Write
   the answer as if it were the first and only version — natural prose from
   the reader's perspective, with zero meta-commentary about revisions.
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

        if not text:
            return text

        cleaned = text

        cleaned = re.sub(r'```(?:json)?\s*[\s\S]*?```', '', cleaned, flags=re.IGNORECASE)


        smart_quote_map = {
            '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
        }
        for bad, good in smart_quote_map.items():
            cleaned = cleaned.replace(bad, good)

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
                        is_data_dump = True
                    except (json.JSONDecodeError, ValueError):

                        kv_pattern_count = len(
                            re.findall(r'"[^"\n]{1,60}"\s*:\s*', candidate)
                        )
                        if kv_pattern_count >= 3:
                            is_data_dump = True

                    if is_data_dump:
                        i = end_idx + 1
                        continue


            out_chars.append(ch)
            i += 1

        cleaned = ''.join(out_chars)

        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        return cleaned if cleaned else text.strip()

    def _generate_initial(self, query: str, context: str, chat_history: list) -> str:
        formatted = self.answer_prompt.format_messages(query=query, context=context, chat_history=chat_history)
        if self.rate_limiter is not None:
            combined_text = " ".join(str(m.content) for m in formatted)
            self.rate_limiter.wait_if_needed(
                TPMRateLimiter.estimate_tokens(combined_text),
                label="Answer Generation"
            )
        response = self.llm.invoke(formatted)
        return self._strip_json_artifacts(response.content.strip())

    def _refine(self, query: str, context: str, previous_answer: str, verification: Dict[str, Any]) -> str:
        missing_refs_text = ", ".join(verification.get("missing_refs") or []) or "None specified"
        formatted = self.refine_prompt.format_messages(
            query=query,
            context=context,
            previous_answer=previous_answer,
            rating=verification.get("rating", 0),
            missing_refs=missing_refs_text,
            notes=verification.get("notes", ""),
        )
        if self.rate_limiter is not None:
            combined_text = " ".join(str(m.content) for m in formatted)
            self.rate_limiter.wait_if_needed(
                TPMRateLimiter.estimate_tokens(combined_text),
                label="Refinement"
            )
        response = self.llm.invoke(formatted)
        return self._strip_json_artifacts(response.content.strip())

    _HAS_DIGIT_RE = re.compile(r'[0-9\u0660-\u0669]')

    def run(self, query: str, context: str, chat_history: list, source_texts: List[str]) -> Dict[str, Any]:
        try:
            _t = time.time()
            answer = self._generate_initial(query, context, chat_history)
            print(f"⏱️   ├─ Initial Answer Generation: {time.time() - _t:.2f}s")
        except Exception as e:
            print(f"⚠️ Answer generation error: {e}")
            return {
                "answer": "We were unable to generate a response at this time. Please try again shortly.",
                "verification": {"rating": 0, "passed": False, "missing_refs": [], "notes": "Generation failed."},
                "attempts_made": 0,
                "self_refine_converged": False,
            }

        if not self._HAS_DIGIT_RE.search(answer):
            print("⏱️   ├─ Verification: تخطّي (الإجابة بدون أرقام)")
            verification = {
                "rating": 8,
                "passed": True,
                "missing_refs": [],
                "notes": "No numeric figures in the answer; verification skipped.",
            }
            return {
                "answer": self._strip_json_artifacts(answer),
                "verification": verification,
                "attempts_made": 0,
                "self_refine_converged": True,
            }

        attempts = []
        total_rounds = self.max_refinement_attempts + 1

        for round_idx in range(total_rounds):
            _t = time.time()
            verification = self.verifier.verify_answer(query, answer, source_texts)
            print(f"⏱️   ├─ Verification round {round_idx + 1}: {time.time() - _t:.2f}s (rating={verification['rating']}, passed={verification['passed']})")
            attempts.append({"answer": answer, "verification": verification})

            if verification["passed"]:
                break

            is_last_round = (round_idx == total_rounds - 1)
            if is_last_round:
                break

            try:
                _t = time.time()
                answer = self._refine(query, context, answer, verification)
                print(f"⏱️   ├─ Refinement round {round_idx + 1}: {time.time() - _t:.2f}s")
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

    def __init__(self, vector_db, llm, adaptive_depth: Optional[AdaptiveRetrievalDepth] = None):
        self.vector_db = vector_db
        self.llm = llm
        self.adaptive_depth = adaptive_depth or AdaptiveRetrievalDepth(
            k_min=4,
            k_upper_bound=25,
            corpus_divisor=10,
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

    VIZ_KEYWORDS = ["chart", "visualiz", "plot", "graph", "draw", "pie", "bar", "line", "trend"]

    FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
    MAIN_MODEL = os.getenv("GROQ_MAIN_MODEL", "openai/gpt-oss-20b")

    def __init__(self, vector_db, chunks: Optional[List[Any]] = None):
        self.vector_db = vector_db
        self.chunks = chunks
        self._hybrid_retriever = HybridRetriever(vector_db, chunks) if chunks else None

        api_key = os.getenv("GROQ_API_KEY")
        secret_key = SecretStr(api_key) if api_key else None

       
        self.llm = ChatGroq(
            model=self.MAIN_MODEL,
            api_key=secret_key,
            temperature=0,
            max_retries=1,
            request_timeout=60,
        )
        self.fast_llm = ChatGroq(
            model=self.FAST_MODEL,
            api_key=secret_key,
            temperature=0,
            max_retries=1,
            request_timeout=30,
        )

        self._main_model_limiter = TPMRateLimiter(
            tpm_limit=int(os.getenv("GROQ_MAIN_MODEL_TPM_LIMIT", "8000")),
            safety_margin=0.9,
        )

     
        self._fast_model_limiter = TPMRateLimiter(
            tpm_limit=int(os.getenv("GROQ_FAST_MODEL_TPM_LIMIT", "30000")),
            safety_margin=0.9,
        )

    def process_query(self, query: str, chat_history: list = None) -> dict:
        _t0 = time.time()

        def _lap(label):
            nonlocal _t0
            now = time.time()
            print(f"⏱️ {label}: {now - _t0:.2f}s")
            _t0 = now

        query_expander = QueryExpander(self.fast_llm) if self._hybrid_retriever else None

        retriever = CRAGRetriever(
            self.vector_db,
            self.fast_llm,
            hybrid_retriever=self._hybrid_retriever,
            query_expander=query_expander,
        )
        relevant_results = retriever.get_relevant_documents(query)
        _lap("Retrieval TOTAL (expansion + hybrid search + CRAG grading)")

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

 
        print(f"📄 Context المرسل للموديل ({len(context)} حرف من {len(relevant_docs)} مستند):")
        print(f"   {context[:300]}{'...' if len(context) > 300 else ''}")

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

        if chat_history is None:
            chat_history = []

        # ========================================================================
    
        # ========================================================================
        refine_engine = SelfRefiningAnswerEngine(
            self.llm,
            verifier_llm=self.fast_llm,
            max_refinement_attempts=1, pass_threshold=7,
            rate_limiter=self._main_model_limiter,          # للموديل الرئيسي فقط (Generation + Refinement)
            verifier_rate_limiter=self._fast_model_limiter,  # منفصل تماماً لـ fast_llm
        )

        needs_chart = any(kw in query.lower() for kw in self.VIZ_KEYWORDS)

        with ThreadPoolExecutor(max_workers=2) as executor:
            answer_future = executor.submit(
                refine_engine.run,
                query=query,
                context=context,
                chat_history=chat_history,
                source_texts=[doc.page_content for doc in relevant_docs],
            )

            chart_future = None
            if needs_chart:
                chart_future = executor.submit(self._build_chart, query)

            refine_result = answer_future.result()
            chart_data = chart_future.result() if chart_future is not None else None
        _lap(f"Answer generation + verification (attempts={refine_result['attempts_made']})" + (" + chart (parallel)" if needs_chart else ""))

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

        return {
            "answer": answer,
            "source_pages": source_pages,
            "confidence": confidence,
            "verification": verification,
            "relevant_docs_count": len(relevant_docs),
            "source_texts": [doc.page_content for doc in relevant_docs],
            "chart": chart_data,
            "self_refine_attempts": refine_result["attempts_made"],
            "self_refine_converged": refine_result["self_refine_converged"],
        }

    def _build_chart(self, query: str) -> Optional[Dict[str, Any]]:
        try:
            _chart_t0 = time.time()
            extractor = FinancialDataExtractor(self.vector_db, self.fast_llm)
            df = extractor.extract_data_from_query(query)
            print(f"⏱️ Chart data extraction: {time.time() - _chart_t0:.2f}s")
            print(f"📊 DataFrame for visualization:\n{df.head()}")

            if df.empty:
                return None

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

            return {
                "success": True,
                "chart": fig.to_json(),
                "title": query,
                "data_preview": df.to_dict(orient="records")
            }
        except Exception as e:
            print(f"⚠️ Chart generation error: {e}")
            return {"success": False, "error": str(e)}

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
