import os
import re
from typing import List, Dict, Any, Literal, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json


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
        
        print(f"🔍 Verification LLM Response: {response.content}")

        rating = 8
        try:
            rating_match = re.search(r'(\d+)(?:/10)?', response.content)
            if rating_match:
                rating = int(rating_match.group(1))
        except Exception as e:
            print(f"⚠️ Error parsing verification rating: {e}")

        return {
            "rating": rating,
            "passed": rating >= 7,
            "notes": response.content if response.content else "No verification notes available."
        }


# ============================================================================
# 3. VISUALIZATION TOOL - Financial Data Visualization
# ============================================================================

class FinancialDataExtractor:
    """Extract financial data from documents for visualization"""
    
    def __init__(self, vector_db):
        self.vector_db = vector_db
    
    def extract_data_from_query(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Extract numerical data from documents based on query
        
        Args:
            query: Natural language query describing the data needed
            k: Number of documents to retrieve
        
        Returns:
            DataFrame with extracted data
        """
        docs = self.vector_db.similarity_search(query, k=k)
        
        all_data = []
        
        for doc in docs:
            # Check if document is from Excel
            if doc.metadata.get('sheet_name'):
                data = self._parse_excel_content(doc.page_content)
                if data:
                    all_data.extend(data)
            else:
                data = self._parse_pdf_tables(doc.page_content)
                if data:
                    all_data.extend(data)
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        return df
    
    def _parse_excel_content(self, content: str) -> List[Dict]:
        """Parse Excel sheet content into structured data"""
        lines = content.split('\n')
        data = []
        
        headers = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            if headers is None and len(parts) > 1:
                headers = parts
            elif headers and len(parts) == len(headers):
                row_dict = dict(zip(headers, parts))
                data.append(row_dict)
        
        return data
    
    def _parse_pdf_tables(self, content: str) -> List[Dict]:
        """Parse tables from PDF text content"""
        lines = content.split('\n')
        data = []
        
        for line in lines:
            if ':' in line and any(char.isdigit() for char in line):
                parts = line.split(':')
                if len(parts) == 2:
                    label = parts[0].strip()
                    value = parts[1].strip()
                    data.append({"label": label, "value": value})
            
            elif '|' in line and any(char.isdigit() for char in line):
                parts = line.split('|')
                if len(parts) >= 2:
                    label = parts[0].strip()
                    value = parts[1].strip()
                    data.append({"label": label, "value": value})
        
        return data


class ChartGenerator:
    """Generate Plotly charts from data"""
    
    @staticmethod
    def create_line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """Create a line chart"""
        fig = px.line(df, x=x, y=y, title=title, markers=True, template="plotly_white")
        fig.update_layout(hovermode='x unified', showlegend=True, height=500)
        return fig
    
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """Create a bar chart"""
        fig = px.bar(df, x=x, y=y, title=title, template="plotly_white")
        fig.update_layout(showlegend=True, height=500)
        return fig
    
    @staticmethod
    def create_pie_chart(df: pd.DataFrame, names: str, values: str, title: str) -> go.Figure:
        """Create a pie chart"""
        fig = px.pie(df, names=names, values=values, title=title, template="plotly_white")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def create_scatter_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """Create a scatter chart"""
        fig = px.scatter(df, x=x, y=y, title=title, template="plotly_white", trendline="ols")
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def create_area_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """Create an area chart"""
        fig = px.area(df, x=x, y=y, title=title, template="plotly_white")
        fig.update_layout(height=500)
        return fig


# ============================================================================
# 4. Agentic RAG - Intelligent Agent System with Visualization
# ============================================================================

class FinancialRAGAgent:
    """
    Intelligent RAG agent that combines:
    - Strategic Financial Analysis (CRAG)
    - Self-Verification (Self-RAG)
    - Financial Data Visualization
    """
    
    def __init__(self, vector_db, use_self_rag: bool = False):
        """
        Initialize the agent with a vector database and an LLM.
        
        Args:
            vector_db: Vector database containing financial documents
            use_self_rag: Enable self-verification (increases accuracy but slower)
        """
        from pydantic import SecretStr
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=SecretStr(api_key) if api_key else None,
            temperature=0
        )
        
        self.vector_db = vector_db
        self.crag_retriever = CRAGRetriever(vector_db, self.llm)
        self.self_rag = SelfRAGVerifier(self.llm) if use_self_rag else None
        self.data_extractor = FinancialDataExtractor(vector_db)
        self.chart_generator = ChartGenerator()
        
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Strategic Financial Analyst with visualization capabilities.
            
            Use the provided context to answer the query with strategic insights.
            
            **When user requests visualization:**
            - Respond with: "I'll create a [chart_type] visualization for you."
            - Then provide the answer with the data context
            
            **Visualization Keywords:**
            - draw, plot, chart, show, visualize, graph, display
            - Chart types: line (trends), bar (comparisons), pie (proportions), scatter (correlations), area (cumulative)
            
            **Strict Guidelines:**
            1. **Accuracy**: Use exact figures and dates from the text
            2. **Strategic Insight**: Provide recommendations based on trends
            3. **Formatting**: Space between numbers and currencies (e.g., 50.5 million SAR)
            4. **Citations**: Mention page numbers [Page X]
            5. **Predictive Analysis**: Explain forecasts using historical patterns
            
            Context:
            {context}
            """),
            ("human", "{question}")
        ])

    def _detect_visualization_request(self, question: str) -> Optional[Dict[str, str]]:
        """
        Detect if the question is asking for a visualization
        
        Returns:
            Dict with chart_type and data_query, or None if not a viz request
        """
        question_lower = question.lower()
        
        # Check for visualization keywords
        viz_keywords = ['draw', 'plot', 'chart', 'show', 'visualize', 'graph', 'display', 'create']
        if not any(keyword in question_lower for keyword in viz_keywords):
            return None
        
        # Detect chart type
        chart_type = "bar"  # default
        
        if any(word in question_lower for word in ["line", "trend", "over time", "growth", "history"]):
            chart_type = "line"
        elif any(word in question_lower for word in ["pie", "proportion", "breakdown", "distribution", "share"]):
            chart_type = "pie"
        elif any(word in question_lower for word in ["scatter", "correlation", "relationship"]):
            chart_type = "scatter"
        elif any(word in question_lower for word in ["area", "cumulative"]):
            chart_type = "area"
        
        return {
            "chart_type": chart_type,
            "data_query": question
        }

    def _create_visualization(self, chart_type: str, data_query: str, 
                            x_axis: str = None, y_axis: str = None, 
                            title: str = None) -> Dict[str, Any]:
        """
        Create a visualization from the query
        
        Args:
            chart_type: Type of chart to create
            data_query: Query to extract data
            x_axis: X-axis column name
            y_axis: Y-axis column name
            title: Chart title
        
        Returns:
            Dictionary with chart data and metadata
        """
        try:
            # Extract data
            df = self.data_extractor.extract_data_from_query(data_query)
            
            if df.empty:
                return {
                    "success": False,
                    "error": "No data found for visualization",
                    "chart": None
                }
            
            # Auto-detect columns if not specified
            if not x_axis and len(df.columns) > 0:
                x_axis = df.columns[0]
            if not y_axis and len(df.columns) > 1:
                y_axis = df.columns[1]
            
            # Generate title if not specified
            if not title:
                title = f"{chart_type.title()} Chart: {data_query}"
            
            # Create chart
            if chart_type == "line":
                fig = self.chart_generator.create_line_chart(df, x_axis, y_axis, title)
            elif chart_type == "bar":
                fig = self.chart_generator.create_bar_chart(df, x_axis, y_axis, title)
            elif chart_type == "pie":
                fig = self.chart_generator.create_pie_chart(df, x_axis, y_axis, title)
            elif chart_type == "scatter":
                fig = self.chart_generator.create_scatter_chart(df, x_axis, y_axis, title)
            elif chart_type == "area":
                fig = self.chart_generator.create_area_chart(df, x_axis, y_axis, title)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported chart type: {chart_type}",
                    "chart": None
                }
            
            # Convert to JSON
            chart_json = fig.to_json()
            
            return {
                "success": True,
                "chart_type": chart_type,
                "chart": chart_json,
                "data_preview": df.head(10).to_dict('records'),
                "data_shape": df.shape,
                "title": title
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "chart": None
            }

    def _format_docs_with_pages(self, graded_results: List[Dict]) -> str:
        """Format filtered documents with their source page numbers"""
        formatted = []
        for item in graded_results:
            doc = item["document"]
            page = doc.metadata.get("page", "Unknown")
            sheet = doc.metadata.get("sheet_name", None)
            
            source_ref = f"Page {page}" if not sheet else f"Sheet: {sheet}"
            formatted.append(f"[Source: {source_ref}]\n{doc.page_content}\n")
        return "\n---\n".join(formatted)

    def process_query(self, question: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Main pipeline: Detect Visualization -> Strategic Retrieval -> Generation -> Self-Verification
        
        Args:
            question: The financial query to answer
            max_retries: Number of retry attempts if verification fails
            
        Returns:
            Dictionary with answer, sources, confidence, verification, and optional chart
        """
        
        print(f"\n{'='*60}")
        print(f"🕵️ Strategic Analysis for: {question}")
        print(f"{'='*60}")

        # Step 0: Check for visualization request
        viz_request = self._detect_visualization_request(question)
        chart_data = None
        
        if viz_request:
            print(f"📊 Visualization requested: {viz_request['chart_type']} chart")
            chart_data = self._create_visualization(
                chart_type=viz_request['chart_type'],
                data_query=viz_request['data_query']
            )
            
            if chart_data.get('success'):
                print(f"✅ Chart created successfully: {chart_data['data_shape']} data points")

        # Step 1: Strategic Retrieval & Grading
        relevant_graded_results = self.crag_retriever.get_relevant_documents(
            question=question, 
            k=5
        )
        
        if not relevant_graded_results:
            return {
                "answer": "The strategic analysis could not find enough relevant data in the reports to answer this specific query.",
                "source_pages": [],
                "confidence": "low",
                "verification": None,
                "relevant_docs_count": 0,
                "chart": chart_data
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
        
        source_pages = sorted(set([
            str(item["document"].metadata.get("page", "Unknown"))
            for item in relevant_graded_results
            if item["document"].metadata.get("page")
        ]), key=lambda x: (x == "Unknown", x))

        # Step 3: Self-Verification
        verification = None
        if self.self_rag:
            print("🔍 Verifying strategic accuracy...")
            verification = self.self_rag.verify_answer(
                question=question,
                answer=answer,
                sources=[item["document"].page_content[:300] for item in relevant_graded_results]
            )
            
            if not verification["passed"] and max_retries > 0:
                print(f"⚠️ Verification score: {verification['rating']}/10 - Retrying...")
                return self.process_query(question, max_retries - 1)
            
            print(f"✅ Verification passed: {verification['rating']}/10")

        # Post-processing
        answer = re.sub(r'(\d)(billion|million|SAR|USD|ريال)', r'\1 \2', answer)

        return {
            "answer": answer,
            "source_pages": source_pages,
            "confidence": "high" if not self.self_rag or (verification and verification["passed"]) else "medium",
            "verification": verification,
            "relevant_docs_count": len(relevant_graded_results),
            "chart": chart_data  # New: includes chart data if visualization requested
        }


# ============================================================================
# 5. Main Factory Function
# ============================================================================

def create_rag(vector_db, use_self_rag: bool = False):
    """
    Create an advanced RAG agent with strategic analysis and visualization
    
    Args:
        vector_db: Vector database containing financial documents
        use_self_rag: Enable Self-RAG for verification
    
    Returns:
        FinancialRAGAgent: Enhanced agent with visualization capabilities
    """
    return FinancialRAGAgent(vector_db, use_self_rag=use_self_rag)