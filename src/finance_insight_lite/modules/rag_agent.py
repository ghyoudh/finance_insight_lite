
import os
from typing import List, Dict, Any, Literal
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field


# ============================================================================
# 1. CRAG - Corrective RAG Component
# ============================================================================

class RetrievalQualityGrader(BaseModel):
    """Grade the quality of retrieved documents"""
    score: Literal["relevant", "irrelevant"] = Field(
        description="Is the document relevant to the question?"
    )
    confidence: float = Field(
        description="Confidence level in the assessment (0-1)"
    )


class CRAGRetriever:
    """Corrective RAG - Improves retrieval quality"""
    
    def __init__(self, vector_db, llm):
        self.vector_db = vector_db
        self.llm = llm
        self.grader_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at evaluating financial document quality.
            Assess whether the following document is relevant to the question.
            
            Criteria:
            - relevant: The document contains direct information to answer the question
            - irrelevant: The document does not help answer the question
            
            Provide your assessment with a confidence level."""),
            ("human", "Question: {question}\n\nDocument: {document}\n\nAssessment:")
        ])
    
    def grade_document(self, question: str, document: str) -> Dict[str, Any]:
        """Grade a single document"""
        structured_llm = self.llm.with_structured_output(RetrievalQualityGrader)
        grader = self.grader_prompt | structured_llm
        
        result = grader.invoke({
            "question": question,
            "document": document[:500]  # First 500 chars for assessment
        })
        
        return {
            "score": result.score,
            "confidence": result.confidence
        }
    
    def retrieve_with_correction(
        self, 
        question: str, 
        k: int = 5,
        relevance_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Retrieve with correction - re-retrieves if quality is low"""
        
        print(f"\n🔍 CRAG: Starting retrieval for: {question[:50]}...")
        
        # Initial retrieval
        initial_docs = self.vector_db.similarity_search(question, k=k)
        
        # Grade each document
        graded_docs = []
        relevant_count = 0
        
        for doc in initial_docs:
            grade = self.grade_document(question, doc.page_content)
            
            if grade["score"] == "relevant" and grade["confidence"] >= relevance_threshold:
                graded_docs.append({
                    "document": doc,
                    "grade": grade,
                    "relevant": True
                })
                relevant_count += 1
            else:
                graded_docs.append({
                    "document": doc,
                    "grade": grade,
                    "relevant": False
                })
        
        print(f"✅ CRAG: Found {relevant_count}/{len(initial_docs)} relevant documents")
        
        # If few relevant documents, re-retrieve with higher k
        if relevant_count < 2:
            print("⚠️ CRAG: Low quality, re-retrieving with higher k...")
            additional_docs = self.vector_db.similarity_search(question, k=k*2)
            
            for doc in additional_docs[k:]:  # Only new documents
                grade = self.grade_document(question, doc.page_content)
                if grade["score"] == "relevant":
                    graded_docs.append({
                        "document": doc,
                        "grade": grade,
                        "relevant": True
                    })
        
        # Sort by relevance and confidence
        graded_docs.sort(
            key=lambda x: (x["relevant"], x["grade"]["confidence"]),
            reverse=True
        )
        
        return graded_docs[:k]  # Return top k documents


# ============================================================================
# 2. Self-RAG - Self-verification
# ============================================================================

class SelfRAGVerifier:
    """Self-verification of answer quality"""
    
    def __init__(self, llm):
        self.llm = llm
        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial auditor.
            Evaluate the following answer based on:
            1. Accuracy: Is the information correct?
            2. Completeness: Does it fully answer the question?
            3. Credibility: Is it supported by sources?
            
            Provide a rating from 1-10 and your notes."""),
            ("human", """Question: {question}
            
Answer: {answer}

Sources used: {sources}

Evaluation:""")
        ])
    
    def verify_answer(
        self, 
        question: str, 
        answer: str, 
        sources: List[str]
    ) -> Dict[str, Any]:
        """Verify answer quality"""
        
        result = (self.verification_prompt | self.llm | StrOutputParser()).invoke({
            "question": question,
            "answer": answer,
            "sources": "\n".join([f"- {s[:100]}..." for s in sources])
        })
        
        return {
            "verification": result,
            "passed": "8" in result or "9" in result or "10" in result
        }


# ============================================================================
# 3. Agentic RAG - Intelligent Agent System
# ============================================================================

class FinancialRAGAgent:
    """Intelligent RAG agent combining CRAG and Self-RAG"""
    
    def __init__(self, vector_db, use_self_rag: bool = False):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
        
        # System components
        self.crag_retriever = CRAGRetriever(vector_db, self.llm)
        self.self_rag = SelfRAGVerifier(self.llm) if use_self_rag else None
        
        # Answer generation prompt
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial analyst.
            Use the provided documents to answer the question accurately.
            
            Important rules:
            - Use numbers exactly as they appear in the documents
            - Add space between numbers and currencies (e.g., 104.9 billion SAR)
            - Mention the page number at the end of your answer
            - If you don't find the answer, state that clearly
            
            Available documents:
            {context}
            """),
            ("human", "{question}")
        ])
    
    def _format_docs_with_pages(self, graded_docs: List[Dict]) -> str:
        """Format documents with page numbers"""
        formatted = []
        for item in graded_docs:
            if item["relevant"]:
                doc = item["document"]
                page = doc.metadata.get("page", "Unknown")
                formatted.append(
                    f"[Page {page}]\n{doc.page_content}\n"
                )
        return "\n---\n".join(formatted)
    
    def process_query(
        self, 
        question: str,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """Process query using Agentic RAG + CRAG + Self-RAG"""
        
        print(f"\n{'='*60}")
        print(f"🤖 Processing question: {question}")
        print(f"{'='*60}")
        
        # Step 1: Retrieval with CRAG
        graded_docs = self.crag_retriever.retrieve_with_correction(
            question=question,
            k=5,
            relevance_threshold=0.6
        )
        
        # Check for relevant documents
        relevant_docs = [d for d in graded_docs if d["relevant"]]
        
        if not relevant_docs:
            return {
                "answer": "Sorry, I couldn't find relevant information to answer your question in the available reports.",
                "source_pages": [],
                "confidence": "low",
                "verification": None
            }
        
        # Step 2: Generate answer
        print(f"\n💡 Generating answer using {len(relevant_docs)} documents...")
        
        context = self._format_docs_with_pages(graded_docs)
        
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.answer_prompt
            | self.llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(question)
        
        # Extract source pages
        source_pages = sorted(set([
            d["document"].metadata.get("page", "Unknown")
            for d in relevant_docs
        ]))
        
        # Step 3: Self-verification (if enabled)
        verification = None
        if self.self_rag:
            print("\n🔍 Self-RAG: Verifying answer quality...")
            verification = self.self_rag.verify_answer(
                question=question,
                answer=answer,
                sources=[d["document"].page_content[:200] for d in relevant_docs]
            )
            
            # Retry if verification failed
            if not verification["passed"] and max_retries > 0:
                print("⚠️ Self-RAG: Answer failed verification, retrying...")
                return self.process_query(question, max_retries - 1)
        
        # Format final answer
        import re
        answer = re.sub(r'(\d)(billion|million|SAR|ريال)', r'\1 \2', answer)
        
        print(f"\n✅ Processing completed successfully!")
        
        return {
            "answer": answer,
            "source_pages": source_pages,
            "confidence": "high" if len(relevant_docs) >= 3 else "medium",
            "verification": verification,
            "relevant_docs_count": len(relevant_docs)
        }


# ============================================================================
# 4. Main Factory Function
# ============================================================================

def create_advanced_rag_agent(vector_db, use_self_rag: bool = False):
    """
    Create an advanced RAG agent
    
    Args:
        vector_db: Vector database
        use_self_rag: Enable Self-RAG for verification (increases accuracy but slower)
    
    Returns:
        FinancialRAGAgent: Ready-to-use RAG agent
    """
    return FinancialRAGAgent(vector_db, use_self_rag=use_self_rag)