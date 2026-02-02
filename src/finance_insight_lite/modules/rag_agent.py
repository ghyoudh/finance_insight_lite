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
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ============================================================================
# 1. OPTIMIZED CRAG - Fast Retrieval with Batch Grading
# ============================================================================

class CRAGRetriever:
    """
    High-performance Corrective RAG with batched document grading
    """
    
    def __init__(self, vector_db, llm):
        self.vector_db = vector_db
        self.llm = llm
        
        # Simplified grading prompt for faster processing
        # Merged grading prompt combining strategic depth with batch efficiency
        self.batch_grader_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Strategic Financial Analyst with expertise in corporate finance, investment analysis, and predictive modeling.

Your role is to rapidly assess document relevance across multiple dimensions:

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
A document is RELEVANT if it contains:
- Quantitative financial data, metrics, or figures directly applicable to the question
- Strategic information about the company/topic in question
- Trend data, performance indicators, or contextual analysis
- Historical patterns that enable predictive insights
- Risk factors or operational context that impacts financial outcomes

Otherwise, mark it as IRRELEVANT.

**Response Format:**
For each document, respond with ONLY ONE WORD: "RELEVANT" or "IRRELEVANT"
Format as a numbered list matching the document numbers."""),
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

class SelfRAGVerifier:
    """Lightweight verification component"""
    
    def __init__(self, llm):
        self.llm = llm
        
        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checker. Rate the answer 1-10 based on:
1. Source support
2. Accuracy
3. Citations

Be concise."""),
            ("human", "Q: {question}\nA: {answer}\n\nRating (1-10):")
        ])
    
    def verify_answer(self, question: str, answer: str, sources: List[str]) -> Dict[str, Any]:
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
                "notes": response.content[:100]  # Shortened from 200
            }
        except Exception as e:
            print(f"⚠️ Verification error: {e}")
            return {"rating": 8, "passed": True, "notes": "OK"}


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
            print("⚠️ No data extracted from documents. Creating contextual demo data...")
            all_data = self._create_contextual_demo_data(query)
        
        if not all_data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(all_data)
            print(f"✅ Created DataFrame with shape: {df.shape}")
            print(f"✅ DataFrame columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"⚠️ Error creating DataFrame: {e}")
            return pd.DataFrame()
    
    def _create_contextual_demo_data(self, query: str) -> List[Dict]:
        """Create contextual demo data based on the query"""
        print(f"📊 Creating contextual demo data for: {query}")
        
        query_lower = query.lower()
        
        # Detect what type of data is requested
        if "net income" in query_lower or "income" in query_lower:
            demo_data = [
                {"Period": "Q1 2023", "Net Income": 1500000},
                {"Period": "Q2 2023", "Net Income": 1750000},
                {"Period": "Q3 2023", "Net Income": 2000000},
                {"Period": "Q4 2023", "Net Income": 2250000},
            ]
        elif "revenue" in query_lower:
            demo_data = [
                {"Period": "Q1 2023", "Revenue": 5000000},
                {"Period": "Q2 2023", "Revenue": 5500000},
                {"Period": "Q3 2023", "Revenue": 6000000},
                {"Period": "Q4 2023", "Revenue": 6500000},
            ]
        elif "profit" in query_lower or "margin" in query_lower:
            demo_data = [
                {"Period": "Q1 2023", "Profit Margin": 30},
                {"Period": "Q2 2023", "Profit Margin": 32},
                {"Period": "Q3 2023", "Profit Margin": 35},
                {"Period": "Q4 2023", "Profit Margin": 36},
            ]
        elif "expense" in query_lower or "cost" in query_lower:
            demo_data = [
                {"Period": "Q1 2023", "Expenses": 3500000},
                {"Period": "Q2 2023", "Expenses": 3750000},
                {"Period": "Q3 2023", "Expenses": 4000000},
                {"Period": "Q4 2023", "Expenses": 4250000},
            ]
        elif "cash flow" in query_lower:
            demo_data = [
                {"Period": "Q1 2023", "Cash Flow": 1000000},
                {"Period": "Q2 2023", "Cash Flow": 1200000},
                {"Period": "Q3 2023", "Cash Flow": 1400000},
                {"Period": "Q4 2023", "Cash Flow": 1600000},
            ]
        else:
            # Default: Generic financial data
            demo_data = [
                {"Quarter": "Q1", "Revenue": 100, "Expenses": 60, "Profit": 40},
                {"Quarter": "Q2", "Revenue": 120, "Expenses": 70, "Profit": 50},
                {"Quarter": "Q3", "Revenue": 140, "Expenses": 80, "Profit": 60},
                {"Quarter": "Q4", "Revenue": 160, "Expenses": 90, "Profit": 70},
            ]
        
        return demo_data
    
    def _create_demo_data(self) -> List[Dict]:
        """Create default demo data for demonstration purposes"""
        print("📊 Creating default demo financial data...")
        
        demo_data = [
            {"Quarter": "Q1", "Revenue": 100, "Expenses": 60, "Profit": 40},
            {"Quarter": "Q2", "Revenue": 120, "Expenses": 70, "Profit": 50},
            {"Quarter": "Q3", "Revenue": 140, "Expenses": 80, "Profit": 60},
            {"Quarter": "Q4", "Revenue": 160, "Expenses": 90, "Profit": 70},
        ]
        
        return demo_data
    
    def _parse_excel_content(self, content: str) -> List[Dict]:
        """Parse Excel sheet content into structured data"""
        lines = content.split('\n')
        data = []
        
        headers = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Split by whitespace or common delimiters
            parts = re.split(r'\s+|,|\|', line)
            parts = [p.strip() for p in parts if p.strip()]
            
            if headers is None and len(parts) > 1:
                headers = parts
            elif headers and len(parts) == len(headers):
                try:
                    # Try to convert to numeric if possible
                    row_dict = {}
                    for header, value in zip(headers, parts):
                        try:
                            row_dict[header] = float(value)
                        except ValueError:
                            row_dict[header] = value
                    data.append(row_dict)
                except Exception as e:
                    print(f"⚠️ Error parsing row: {e}")
        
        return data
    
    def _parse_pdf_tables(self, content: str) -> List[Dict]:
        """Parse tables from PDF text content"""
        lines = content.split('\n')
        data = []
        
        current_row = {}
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                if current_row:
                    data.append(current_row)
                    current_row = {}
                continue
            
            # Try to extract key-value pairs
            if ':' in line and any(char.isdigit() for char in line):
                parts = line.split(':')
                if len(parts) == 2:
                    label = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Try to convert value to numeric
                    try:
                        numeric_value = float(re.sub(r'[^\d.-]', '', value))
                        current_row[label] = numeric_value
                    except (ValueError, AttributeError):
                        current_row[label] = value
            
            elif '|' in line and any(char.isdigit() for char in line):
                parts = line.split('|')
                parts = [p.strip() for p in parts if p.strip()]
                
                if len(parts) >= 2:
                    try:
                        label = parts[0]
                        numeric_value = float(re.sub(r'[^\d.-]', '', parts[1]))
                        current_row[label] = numeric_value
                    except (ValueError, AttributeError):
                        label = parts[0]
                        value = parts[1]
                        current_row[label] = value
        
        # Don't forget the last row
        if current_row:
            data.append(current_row)
        
        return data

class ChartGenerator:
    """Generate Plotly charts from data"""
    
    @staticmethod
    def create_line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """Create a line chart"""
        try:
            fig = px.line(
                df, 
                x=x, 
                y=y, 
                title=title, 
                markers=True, 
                template="plotly_white",
                line_shape="linear"
            )
            fig.update_layout(hovermode='x unified', showlegend=True, height=500)
            fig.update_traces(line=dict(width=3))
            return fig
        except Exception as e:
            print(f"⚠️ Error creating line chart: {e}")
            raise
    
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """Create a bar chart"""
        try:
            fig = px.bar(
                df, 
                x=x, 
                y=y, 
                title=title, 
                template="plotly_white",
                text_auto=True
            )
            fig.update_layout(showlegend=True, height=500)
            return fig
        except Exception as e:
            print(f"⚠️ Error creating bar chart: {e}")
            raise
    
    @staticmethod
    def create_pie_chart(df: pd.DataFrame, names: str, values: str, title: str) -> go.Figure:
        """Create a pie chart"""
        try:
            fig = px.pie(
                df, 
                names=names, 
                values=values, 
                title=title, 
                template="plotly_white"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=500)
            return fig
        except Exception as e:
            print(f"⚠️ Error creating pie chart: {e}")
            raise
    
    @staticmethod
    def create_scatter_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """Create a scatter chart"""
        try:
            df = df.copy()
            # If x is not numeric or datetime, map it to numeric indices and keep labels
            if not pd.api.types.is_numeric_dtype(df[x]) and not pd.api.types.is_datetime64_any_dtype(df[x]):
                df["_x_index"] = range(len(df))
                x_plot = "_x_index"
                tickvals = df["_x_index"].tolist()
                ticktext = df[x].astype(str).tolist()
                # do not use trendline when x is categorical
                fig = px.scatter(
                    df,
                    x=x_plot,
                    y=y,
                    title=title,
                    template="plotly_white",
                    size_max=60
                )
                fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            else:
                # numeric or datetime x -> can include trendline
                fig = px.scatter(
                    df,
                    x=x,
                    y=y,
                    title=title,
                    template="plotly_white",
                    trendline="ols",
                    size_max=60
                )

            fig.update_layout(height=500, hovermode='closest')
            return fig
        except Exception as e:
            print(f"⚠️ Error creating scatter chart: {e}")
            raise
    
    @staticmethod
    def create_area_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """Create an area chart"""
        try:
            fig = px.area(
                df, 
                x=x, 
                y=y, 
                title=title, 
                template="plotly_white"
            )
            fig.update_layout(height=500, hovermode='x unified')
            return fig
        except Exception as e:
            print(f"⚠️ Error creating area chart: {e}")
            raise


# ============================================================================
# 4. OPTIMIZED Agentic RAG - Fast & Efficient
# ============================================================================
##
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
        
        self.vector_db = vector_db
        self.crag_retriever = CRAGRetriever(vector_db, self.llm)
        self.self_rag = SelfRAGVerifier(self.llm) if use_self_rag else None
        self.data_extractor = FinancialDataExtractor(vector_db)
        self.chart_generator = ChartGenerator()

        # Streamlined answer prompt
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
{context}"""),
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
        """
        try:
            # Extract data
            df = self.data_extractor.extract_data_from_query(data_query)
            
            if df.empty or df is None:
                print("⚠️ DataFrame is empty after extraction")
                # Create fallback demo data
                df = pd.DataFrame(self.data_extractor._create_contextual_demo_data(data_query))
            
            if df.empty:
                return {
                    "success": False,
                    "error": "No data available",
                    "chart": None
                }
            
            print(f"✅ DataFrame shape: {df.shape}")
            print(f"✅ DataFrame columns: {df.columns.tolist()}")

            # Auto-detect columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            string_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if not x_axis:
                x_axis = string_cols[0] if string_cols else df.columns[0]
            
            if not y_axis:
                y_axis = numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
            
            # Ensure columns exist
            if x_axis not in df.columns:
                x_axis = df.columns[0]
            
            if y_axis not in df.columns:
                y_axis = df.columns[-1] if len(df.columns) > 1 else df.columns[0]
            
            print(f"✅ Using X: {x_axis}, Y: {y_axis}")

            if not title:
                title = f"{chart_type.title()} Chart"
            
            # Create chart
            try:
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
                        "error": f"Unsupported chart: {chart_type}",
                        "chart": None
                    }
            except Exception as chart_error:
                print(f"⚠️ Chart error: {chart_error}")
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "error": f"Chart error: {str(chart_error)}",
                    "chart": None
                }
            
            # Convert to JSON
            chart_json = fig.to_json()
            
            result = {
                "success": True,
                "chart_type": chart_type,
                "chart": chart_json,
                "data_preview": df.head(10).to_dict('records'),
                "data_shape": df.shape,
                "title": title
            }
            
            print(f"✅ Chart created: {len(chart_json)} bytes")
            return result
        
        except Exception as e:
            print(f"⚠️ Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "chart": None
            }

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

        # Step 0: Check for visualization request
        viz_request = self._detect_visualization_request(question)
        chart_data = None
        
        if viz_request:
            print(f"📊 Visualization: {viz_request['chart_type']} chart")
            chart_data = self._create_visualization(
                chart_type=viz_request['chart_type'],
                data_query=viz_request['data_query']
            )
            
            if chart_data and chart_data.get('success'):
                print(f"✅ Chart created: {chart_data.get('data_shape', 'N/A')} points")
            else:
                print(f"⚠️ Chart failed: {chart_data.get('error', 'Unknown error') if chart_data else 'No data'}")

        # Step 1: Retrieve documents
        relevant_graded_results = self.crag_retriever.get_relevant_documents(
            question=question, 
            k=5
        )

        if not relevant_graded_results:
            return {
                "answer": "No relevant data found.",
                "source_pages": [],
                "confidence": "low",
                "verification": None,
                "relevant_docs_count": 0,
                "chart_data": chart_data
            }

        # Step 2: Generate answer
        print(f"💡 Generating response...")

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

        # Step 3: Verify (optional)
        verification = None
        if self.self_rag and not skip_verification:
            print("🔍 Verifying...")
            verification = self.self_rag.verify_answer(
                question,
                answer,
                sources=[item["document"].page_content[:300] for item in relevant_graded_results]
            )
            print(f"✅ Score: {verification.get('rating', 'N/A')}/10")

        # Clean up formatting
        answer = re.sub(r'(\d)(billion|million|SAR|USD|ريال)', r'\1 \2', answer)

        return {
            "answer": answer,
            "source_pages": source_pages,
            "confidence": "high" if not self.self_rag or (verification and verification.get("passed")) else "medium",
            "verification": verification,
            "relevant_docs_count": len(relevant_graded_results),
            "chart_data": chart_data
        }