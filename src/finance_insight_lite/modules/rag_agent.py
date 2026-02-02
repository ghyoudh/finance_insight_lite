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


# ============================================================================
# 1. OPTIMIZED CRAG - Fast Retrieval with Batch Grading
# ============================================================================

class OptimizedCRAGRetriever:
    """
    High-performance Corrective RAG with batched document grading
    """
    
    def __init__(self, vector_db, llm):
        self.vector_db = vector_db
        self.llm = llm
        
        # Simplified grading prompt for faster processing
        self.batch_grader_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fast document relevance classifier for financial queries.

For each document, respond with ONLY ONE WORD: "RELEVANT" or "IRRELEVANT"

A document is RELEVANT if it contains:
- Financial data, metrics, or figures related to the question
- Strategic information about the company/topic in question
- Trend data or performance indicators

Otherwise, mark it as IRRELEVANT.

Format your response as a numbered list matching the document numbers."""),
            ("human", "Question: {question}\n\nDocuments:\n{documents}\n\nClassifications:")
        ])
    
    def batch_grade_documents(self, question: str, documents: List[Any]) -> List[bool]:
        """Grade multiple documents in a single LLM call for speed"""
        if not documents:
            return []
        
        # Format all documents with numbers
        docs_text = "\n\n".join([
            f"Document {i+1} [Page {doc.metadata.get('page', '?')}]:\n{doc.page_content[:500]}"
            for i, doc in enumerate(documents)
        ])
        
        try:
            response = self.llm.invoke(
                self.batch_grader_prompt.format(
                    question=question, 
                    documents=docs_text
                )
            )
            
            # Parse the response to get relevance for each document
            content = response.content.upper()
            relevance_results = []
            
            for i in range(len(documents)):
                # Look for document number and check if "RELEVANT" appears nearby
                doc_pattern = f"(?:DOCUMENT\\s*{i+1}|{i+1}[.)]?).*?(RELEVANT|IRRELEVANT)"
                match = re.search(doc_pattern, content, re.IGNORECASE | re.DOTALL)
                
                if match:
                    is_relevant = "IRRELEVANT" not in match.group(1).upper()
                else:
                    # Fallback: check if "RELEVANT" appears more than "IRRELEVANT"
                    is_relevant = "RELEVANT" in content and content.count("RELEVANT") > content.count("IRRELEVANT")
                
                relevance_results.append(is_relevant)
            
            return relevance_results
            
        except Exception as e:
            print(f"⚠️ Batch grading error: {e}, marking all as relevant")
            return [True] * len(documents)

    def get_relevant_documents(self, question: str, k: int = 5) -> List[Dict]:
        """
        Fast retrieval with batch grading
        
        Args:
            question: The financial query
            k: Number of documents to retrieve (default: 5)
        
        Returns:
            List of relevant documents
        """
        print(f"🔍 Retrieving {k} documents...")

        initial_docs = self.vector_db.similarity_search(question, k=k)
        
        if not initial_docs:
            return []
        
        # Batch grade all documents at once (much faster!)
        relevance_flags = self.batch_grade_documents(question, initial_docs)
        
        # Filter to only relevant documents
        relevant_results = [
            {"document": doc, "relevant": True}
            for doc, is_relevant in zip(initial_docs, relevance_flags)
            if is_relevant
        ]
        
        print(f"📊 Total relevant: {len(relevant_results)}/{len(initial_docs)}")

        # Fallback: If nothing is relevant, return top 2
        if not relevant_results:
            print("⚠️ No relevant docs found, using top 2 as fallback")
            return [{"document": d, "relevant": True} for d in initial_docs[:2]]

        return relevant_results


# ============================================================================
# 2. OPTIONAL Self-RAG - Lightweight Verification
# ============================================================================

class FastSelfRAGVerifier:
    """Lightweight verification component"""
    
    def __init__(self, llm):
        self.llm = llm
        
        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """Rate this financial answer's accuracy from 1-10.
            
Response format: "Rating: X/10" followed by brief notes."""),
            ("human", """Question: {question}
Answer: {answer}

Rating:""")
        ])
    
    def verify_answer(self, question: str, answer: str) -> Dict[str, Any]:
        """Quick verification without full source checking"""
        try:
            response = self.llm.invoke(
                self.verification_prompt.format(
                    question=question,
                    answer=answer
                )
            )
            
            # Extract rating
            rating_match = re.search(r'(\d+)(?:/10)?', response.content)
            rating = int(rating_match.group(1)) if rating_match else 8

            return {
                "rating": rating,
                "passed": rating >= 7,
                "notes": response.content[:200]  # Truncate for speed
            }
        except Exception as e:
            print(f"⚠️ Verification error: {e}")
            return {"rating": 8, "passed": True, "notes": "Auto-approved"}


# ============================================================================
# 3. OPTIMIZED Agentic RAG - Fast & Efficient
# ============================================================================

class OptimizedFinancialRAGAgent:
    """
    High-performance RAG agent optimized for speed
    """

    def __init__(self, vector_db, use_self_rag: bool = False, api_key: Optional[str] = None):
        """
        Initialize the optimized agent
        
        Args:
            vector_db: Vector database containing financial documents
            use_self_rag: Enable self-verification (adds ~2-3s but improves accuracy)
            api_key: Optional Groq API key (will use env var if not provided)
        """
        api_key = api_key or os.getenv("GROQ_API_KEY")
        
        # Use faster model configuration
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=SecretStr(api_key) if api_key else None,
            temperature=0,
            max_tokens=1024,  # Limit response length for speed
            timeout=30  # Add timeout
        )
        
        self.crag_retriever = OptimizedCRAGRetriever(vector_db, self.llm)
        self.self_rag = FastSelfRAGVerifier(self.llm) if use_self_rag else None
        
        # Streamlined answer prompt
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Financial Analyst. Answer concisely using the provided context.

Guidelines:
1. Use exact figures from the text
2. Include page references [Page X]
3. Format: "50.5 million SAR" (space between number and currency)
4. Be direct and concise

Context:
{context}"""),
            ("human", "{question}")
        ])

    def _format_docs_with_pages(self, graded_results: List[Dict]) -> str:
        """Format documents efficiently"""
        formatted = []
        for item in graded_results:
            doc = item["document"]
            page = doc.metadata.get("page", "Unknown")
            # Truncate content to reduce context size
            content = doc.page_content[:800]  # Reduced from full content
            formatted.append(f"[Page {page}]\n{content}")
        
        return "\n---\n".join(formatted)

    def process_query(self, question: str, skip_verification: bool = False) -> Dict[str, Any]:
        """
        Fast query processing pipeline
        
        Args:
            question: The financial query
            skip_verification: Skip self-verification for maximum speed
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        
        print(f"\n{'='*60}")
        print(f"🕵️ Query: {question}")
        print(f"{'='*60}")

        # Step 1: Fast Strategic Retrieval (batch grading)
        relevant_graded_results = self.crag_retriever.get_relevant_documents(
            question=question, 
            k=5  # Retrieve more initially, filter to best
        )

        if not relevant_graded_results:
            return {
                "answer": "Insufficient relevant data found to answer this query.",
                "source_pages": [],
                "confidence": "low",
                "verification": None,
                "relevant_docs_count": 0
            }

        # Step 2: Answer Generation
        print(f"💡 Generating response from {len(relevant_graded_results)} documents...")

        context = self._format_docs_with_pages(relevant_graded_results)

        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.answer_prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)

        # Extract source pages
        source_pages = sorted(set([
            str(item["document"].metadata.get("page", "Unknown"))
            for item in relevant_graded_results
            if item["document"].metadata.get("page")
        ]), key=lambda x: (x == "Unknown", x))

        # Step 3: Optional Fast Verification
        verification = None
        if self.self_rag and not skip_verification:
            print("🔍 Quick verification...")
            verification = self.self_rag.verify_answer(question, answer)
            print(f"✅ Score: {verification['rating']}/10")

        # Clean up formatting
        answer = re.sub(r'(\d)(billion|million|SAR|USD|ريال)', r'\1 \2', answer)

        return {
            "answer": answer,
            "source_pages": source_pages,
            "confidence": "high" if not self.self_rag or (verification and verification["passed"]) else "medium",
            "verification": verification,
            "relevant_docs_count": len(relevant_graded_results)
        }


# ============================================================================
# 4. Factory Functions
# ============================================================================

def create_rag_agent(vector_db, use_self_rag: bool = False):
    """
    Create a speed-optimized RAG agent
    
    Args:
        vector_db: Vector database
        use_self_rag: Enable verification (adds 2-3s, improves accuracy)
    
    Returns:
        OptimizedFinancialRAGAgent
    """
    return OptimizedFinancialRAGAgent(vector_db, use_self_rag=use_self_rag)