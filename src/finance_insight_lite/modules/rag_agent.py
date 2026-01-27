import os
import re
from typing import List, Dict, Any, Literal
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field


# ============================================================================
# 1. CRAG - Corrective RAG Component with Strategic Analysis
# ============================================================================

class CRAGRetriever:
    """
    Corrective RAG with Strategic Financial Analysis
    Improves retrieval quality through intelligent document grading
    """
    
    def __init__(self, vector_db, llm):
        self.vector_db = vector_db
        self.llm = llm
        
        self.grader_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Strategic Financial Analyst with expertise in corporate finance, investment analysis, and predictive modeling.

Your role extends beyond data retrieval to encompass:

**Core Analytical Framework:**
1. **Historical Performance Analysis**
   - Identify trends in revenue, net income, operating costs, and margins
   - Recognize cyclical patterns and anomalies in financial metrics
   - Track year-over-year and quarter-over-quarter performance variations

2. **Predictive Signal Detection**
   - Correlate past strategic decisions (M&A, CapEx, R&D) with subsequent financial outcomes
   - Identify leading indicators that preceded significant performance shifts
   - Assess how market events, policy changes, or corporate announcements impacted results

3. **Strategic Context Evaluation**
   - Understand the business logic behind financial changes
   - Connect operational decisions to financial performance
   - Evaluate risk factors and their potential future impact

**Document Relevance Criteria:**
- **Highly Relevant**: Contains quantitative data, trends, or context directly applicable to answering the question
- **Moderately Relevant**: Provides supporting context or partial data that contributes to analysis
- **Irrelevant**: Lacks financial substance or connection to the analytical requirements

Classify the document as: 'Highly Relevant', 'Moderately Relevant', or 'Irrelevant'."""),
            ("human", "Question: {question}\n\nDocument: {document}\n\nAssessment:")
        ])
    
    def grade_document(self, question: str, document: str) -> Dict[str, Any]:
        """Grade a single document based on strategic relevance"""
        response = self.llm.invoke(
            self.grader_prompt.format(question=question, document=document[:1000])
        )
        return {
            "assessment": response.content, 
            "document": document
        }

    def get_relevant_documents(self, question: str, k: int = 5) -> List[Dict]:
        """
        Retrieve documents, grade them, and filter only the relevant ones
        
        Args:
            question: The financial query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with grades
        """
        print(f"🔍 Retrieving and grading {k} documents...")
        
        initial_docs = self.vector_db.similarity_search(question, k=k)
        relevant_results = []
        
        for doc in initial_docs:
            grade_result = self.grade_document(question, doc.page_content)
            assessment = grade_result["assessment"].lower()
            
            # Check if the LLM marked it as relevant
            if "highly relevant" in assessment or "moderately relevant" in assessment or "relevant" in assessment:
                relevant_results.append({
                    "document": doc,
                    "relevant": True,
                    "grade": grade_result
                })
                print(f"  ✓ Found relevant doc from page {doc.metadata.get('page', '?')}")
        
        print(f"📊 Total relevant: {len(relevant_results)}/{len(initial_docs)}")
        
        # Fallback: If nothing is relevant, return the first 2 docs to avoid error
        if not relevant_results:
            print("⚠️ No highly relevant docs found, using top 2 as fallback")
            return [{"document": d, "relevant": True} for d in initial_docs[:2]]
        
        return relevant_results


# ============================================================================
# 2. Self-RAG - Self-verification Component
# ============================================================================

class SelfRAGVerifier:
    """Self-verification component to validate generated answers"""
    
    def __init__(self, llm):
        self.llm = llm
        
        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a meticulous fact-checker for financial analysis.
            
            Verify if the provided answer is:
            1. Supported by the source documents
            2. Accurate in its numerical claims
            3. Properly cited with page references
            
            Rate the answer on a scale of 1-10 and provide specific notes on any issues."""),
            ("human", """Question: {question}
            
Answer: {answer}

Sources: {sources}

Verification Assessment:""")
        ])
    
    def verify_answer(self, question: str, answer: str, sources: List[str]) -> Dict[str, Any]:
        """Verify the generated answer against source documents"""
        
        sources_text = "\n---\n".join(sources)
        
        response = self.llm.invoke(
            self.verification_prompt.format(
                question=question,
                answer=answer,
                sources=sources_text
            )
        )
        
        # Parse the rating from response
        rating = 8  # Default rating
        try:
            # Try to extract rating number from response
            rating_match = re.search(r'(\d+)(?:/10)?', response.content)
            if rating_match:
                rating = int(rating_match.group(1))
        except:
            pass
        
        return {
            "rating": rating,
            "passed": rating >= 7,
            "notes": response.content
        }


# ============================================================================
# 3. Agentic RAG - Intelligent Agent System
# ============================================================================

class FinancialRAGAgent:
    """
    Intelligent RAG agent that combines Strategic Financial Analysis (CRAG) 
    and Self-Verification (Self-RAG).
    """
    
    def __init__(self, vector_db, use_self_rag: bool = False):
        """
        Initialize the agent with a vector database and an LLM.
        
        Args:
            vector_db: Vector database containing financial documents
            use_self_rag: Enable self-verification (increases accuracy but slower)
        """
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
        
        # Initialize the strategic retriever component
        self.crag_retriever = CRAGRetriever(vector_db, self.llm)
        
        # Initialize the self-verification component if enabled
        self.self_rag = SelfRAGVerifier(self.llm) if use_self_rag else None
        
        # Define the system prompt for generating financial answers
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Strategic Financial Analyst.
            Use the provided context, which has been pre-screened for strategic relevance, to answer the query.
            
            Strict Guidelines:
            1. **Accuracy**: Use exact figures and dates from the text
            2. **Strategic Insight**: Based on trends in the documents, provide strategic recommendations when appropriate
            3. **Formatting**: Ensure a space between numbers and currencies (e.g., 50.5 million SAR)
            4. **Citations**: Explicitly mention the page number [Page X] for every key finding
            5. **Predictive Analysis**: When asked for forecasts, explain the logical basis using historical patterns
            
            Context:
            {context}
            """),
            ("human", "{question}")
        ])

    def _format_docs_with_pages(self, graded_results: List[Dict]) -> str:
        """
        Format filtered documents with their source page numbers for the LLM context.
        
        Args:
            graded_results: List of graded document results
            
        Returns:
            Formatted string with page citations
        """
        formatted = []
        for item in graded_results:
            doc = item["document"]
            page = doc.metadata.get("page", "Unknown")
            sheet = doc.metadata.get("sheet_name", None)
            
            source_ref = f"Page {page}" if not sheet else f"Sheet: {sheet}"
            formatted.append(
                f"[Source: {source_ref}]\n{doc.page_content}\n"
            )
        return "\n---\n".join(formatted)

    def process_query(self, question: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Main pipeline: Strategic Retrieval -> Generation -> Self-Verification
        
        Args:
            question: The financial query to answer
            max_retries: Number of retry attempts if verification fails
            
        Returns:
            Dictionary containing answer, sources, confidence, and verification results
        """
        
        print(f"\n{'='*60}")
        print(f"🕵️ Strategic Analysis for: {question}")
        print(f"{'='*60}")

        # Step 1: Strategic Retrieval & Grading
        relevant_graded_results = self.crag_retriever.get_relevant_documents(
            question=question, 
            k=5
        )
        
        # Handle cases where no relevant strategic data is found
        if not relevant_graded_results:
            return {
                "answer": "The strategic analysis could not find enough relevant data in the reports to answer this specific query.",
                "source_pages": [],
                "confidence": "low",
                "verification": None,
                "relevant_docs_count": 0
            }

        # Step 2: Answer Generation
        print(f"💡 Generating strategic response from {len(relevant_graded_results)} key documents...")
        
        context = self._format_docs_with_pages(relevant_graded_results)
        
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.answer_prompt
            | self.llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(question)
        
        # Extract unique page numbers from the metadata
        source_pages = sorted(list(set([
            item["document"].metadata.get("page", item["document"].metadata.get("sheet_name", "Unknown"))
            for item in relevant_graded_results
        ])))

        # Step 3: Self-Verification (Self-RAG)
        verification = None
        if self.self_rag:
            print("🔍 Verifying strategic accuracy...")
            verification = self.self_rag.verify_answer(
                question=question,
                answer=answer,
                sources=[item["document"].page_content[:300] for item in relevant_graded_results]
            )
            
            # If verification fails, retry the process
            if not verification["passed"] and max_retries > 0:
                print(f"⚠️ Verification score: {verification['rating']}/10 - Retrying...")
                return self.process_query(question, max_retries - 1)
            
            print(f"✅ Verification passed: {verification['rating']}/10")

        # Post-processing: Clean up currency and number formatting
        answer = re.sub(r'(\d)(billion|million|SAR|USD|ريال)', r'\1 \2', answer)

        return {
            "answer": answer,
            "source_pages": source_pages,
            "confidence": "high" if not self.self_rag or (verification and verification["passed"]) else "medium",
            "verification": verification,
            "relevant_docs_count": len(relevant_graded_results)
        }


# ============================================================================
# 4. Main Factory Function
# ============================================================================

def create_advanced_rag_agent(vector_db, use_self_rag: bool = False):
    """
    Create an advanced RAG agent with strategic analysis capabilities
    
    Args:
        vector_db: Vector database containing financial documents
        use_self_rag: Enable Self-RAG for verification (increases accuracy but slower)
    
    Returns:
        FinancialRAGAgent: Ready-to-use RAG agent
    """
    return FinancialRAGAgent(vector_db, use_self_rag=use_self_rag)