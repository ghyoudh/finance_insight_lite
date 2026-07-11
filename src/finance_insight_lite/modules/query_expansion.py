from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ============================================================================
# Query Expansion — يولّد صياغات بديلة للسؤال قبل الاسترجاع
# ============================================================================
#
# الهدف: تقريب صياغة السؤال من صياغة المستندات، عشان يستفيد منه كل من
# BM25 (لفظي) والـ vector search (دلالي) بنفس الوقت — بدل ما نعتمد فقط
# على قوة موديل الـ embedding بالتقاط المعنى.



class ExpandedQueries(BaseModel):
    """قائمة الصياغات البديلة للسؤال الأصلي"""
    queries: List[str] = Field(
        description="2 to 3 alternative phrasings of the original question. "
                    "Each one must preserve the EXACT same intent/meaning as "
                    "the original — do not introduce new questions, do not "
                    "narrow or broaden the scope. Only vary: synonyms, "
                    "formality level, word order, or how the same financial "
                    "concept might be described in a source document (e.g. "
                    "'أداء الشركة المالي بآخر 3 شهور' -> 'إيرادات الربع "
                    "الثالث'). Keep the same language as the original question."
    )


class QueryExpander:


    def __init__(self, llm, num_variants: int = 3):
        self.llm = llm
        self.num_variants = num_variants
        self.structured_llm = self.llm.with_structured_output(ExpandedQueries)

        self.expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a query reformulation assistant for a
financial document retrieval system (Arabic/English).

TASK: Given the user's original question, generate {num_variants} alternative
phrasings that preserve the exact same intent but use different wording —
synonyms, different formality, different word order, or terminology closer
to how a financial report/document might phrase the same concept.

RULES:
- Do NOT change the meaning, scope, or intent of the question.
- Do NOT answer the question — only rephrase it.
- Keep the same language as the original question (Arabic -> Arabic,
  English -> English).
- Each variant must be meaningfully different in wording from the others
  and from the original — not trivial punctuation/spacing changes.
- Return ONLY the variants via the schema, nothing else."""),
            ("human", "Original question: {question}\n\nGenerate the alternative phrasings now.")
        ])

    def expand(self, question: str) -> List[str]:
        """
        يرجّع قائمة تحتوي على السؤال الأصلي + الصياغات البديلة (فريدة،
        بدون تكرار). لو فشل التوليد لأي سبب، يرجّع السؤال الأصلي فقط
        (fail-safe — النظام يستمر يشتغل بدون توسعة بدل ما يتوقف).
        """
        try:
            result: ExpandedQueries = self.structured_llm.invoke(
                self.expansion_prompt.format(question=question)
            )
            variants = [q.strip() for q in result.queries if q and q.strip()]
        except Exception as e:
            print(f"⚠️ Query expansion error: {e} — سيتم الاعتماد على السؤال الأصلي فقط")
            variants = []

        all_queries = [question] + variants

        seen = set()
        deduped = []
        for q in all_queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(q)

        print(f"🔎 توسعة السؤال: {len(deduped)} صياغة (شامل الأصلي)")
        return deduped
