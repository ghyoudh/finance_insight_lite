from typing import Any, Dict, List, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class QueryEvaluation(BaseModel):
    needs_retrieval: bool = Field(
        description="True if answering this query requires document context"
    )
    reason: str = Field(max_length=100, description="One short reason for the decision")


class CoordinatorInstruction(BaseModel):
    action: Literal["ANSWER_DIRECT", "ANSWER_FROM_CONTEXT", "REPORT_NOT_FOUND"]
    context_status: Literal["verified", "not_found", "n/a"]
    note: str = Field(default="", max_length=200)


class WorkflowCoordinator:
    """
    Central orchestrator. Does NOT grade document relevance (CRAG's job) and
    does NOT write the final answer (SelfRefiningAnswerEngine's job). It only:
      1. Decides whether retrieval is needed at all.
      2. Hands off to CRAGRetriever when it is.
      3. Converts CRAG's result into a structured instruction for the generator.
    """

    def __init__(self, fast_llm, crag_retriever, rate_limiter=None):
        self.fast_llm = fast_llm
        self.crag_retriever = crag_retriever
        self.rate_limiter = rate_limiter
        self.structured_eval_llm = fast_llm.with_structured_output(QueryEvaluation)

        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a routing classifier for a financial document Q&A system.

Decide whether answering the user's query requires retrieving information from
the uploaded financial documents, or whether it can be answered without retrieval.

needs_retrieval = False for:
- Greetings, small talk, thanks
- Meta questions about the system itself ("what can you do", "كيف تشتغل")
- Follow-ups fully answerable from Chat History alone, with no new numbers/facts
  needed from the documents

needs_retrieval = True for:
- Any question asking for a number, fact, trend, comparison, or explanation that
  would require checking the financial documents
- Follow-ups asking for a NEW figure or deeper detail not already in Chat History

When in doubt, prefer needs_retrieval = True — a false "no retrieval needed" is
worse than one unnecessary retrieval."""),
            ("human", "Chat History: {chat_history}\n\nQuery: {query}\n\nEvaluate now.")
        ])

    def evaluate(self, query: str, chat_history: List[Any]) -> QueryEvaluation:
        try:
            formatted_prompt = self.eval_prompt.format(query=query, chat_history=chat_history)
            if self.rate_limiter is not None:
                self.rate_limiter.wait_if_needed(
                    self.rate_limiter.estimate_tokens(formatted_prompt),
                    label="Coordinator Evaluation",
                )

            return self.structured_eval_llm.invoke(formatted_prompt)
        except Exception as e:
            print(f"⚠️ Coordinator evaluation error: {e} — defaulting to needs_retrieval=True")
            return QueryEvaluation(needs_retrieval=True, reason="evaluation_failed_safe_default")

    def route(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """Calls CRAG and converts its result into a structured instruction."""
        relevant_results = self.crag_retriever.get_relevant_documents(query, k=k)
        relevant_docs = [r["document"] for r in relevant_results]

        if not relevant_docs:
            return {
                "instruction": CoordinatorInstruction(
                    action="REPORT_NOT_FOUND",
                    context_status="not_found",
                    note="No relevant content found in the documents for this query.",
                ),
                "documents": [],
            }

        return {
            "instruction": CoordinatorInstruction(
                action="ANSWER_FROM_CONTEXT",
                context_status="verified",
                note=f"{len(relevant_docs)} relevant document(s) found via CRAG.",
            ),
            "documents": relevant_docs,
        }
