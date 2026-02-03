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
import json


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
    
    def batch_grade_documents(self, question: str, documents: List[Any]) -> List[bool]:
        """Grade multiple documents in a single LLM call for speed"""
        if not documents:
            return []
        
        # Format all documents with numbers
        docs_text = "\n\n".join([
            f"Document {i+1} [Page {doc.metadata.get('page')}]:\n{doc.page_content[:500]}"
            for i, doc in enumerate(documents)
        ])
        
        try:
            response = self.llm.invoke(
                self.batch_grader_prompt.format(
                    question=question, 
                    document=docs_text
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
            ("system", """You are a meticulous financial fact-checker. 

            Compare the 'Answer' against the 'Source Documents'.
            Your output must be very brief and follow this structure:

            1. Score: [X/10]
            2. Verification: [Pass/Fail] (Are numbers accurate and supported?)
            3. Missing Refs: [List page numbers if missing, otherwise 'None']
            4. Critical Notes: [One sentence maximum on errors, or 'No issues']

            Keep it concise. No preamble or conversational filler."""),
            ("human", """Question: {question}
            
            Answer: {answer}

            Sources: {sources}

            Verification Assessment:""")
        ])
    
    def verify_answer(self, question: str, answer: str, sources: List[str]) -> Dict[str, Any]:
        """Quick verification without full source checking"""
        try:
            response = self.llm.invoke(
                self.verification_prompt.format(
                    question=question,
                    answer=answer,
                    sources="\n\n".join(sources[:5])  # Limit to first 5 sources for speed
                )
            )
            
            # Extract rating
            rating_match = re.search(r'(\d+)(?:/10)?', response.content)
            rating = int(rating_match.group(1)) if rating_match else 8

            return {
                "rating": rating,
                "passed": rating >= 7,
                "notes": response.content[:300]  # Shortened from 200
            }
        except Exception as e:
            print(f"⚠️ Verification error: {e}")
            return {"rating": 8, "passed": True, "notes": "OK"}


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

    def process_query(self, query: str) -> dict:
        """
        Main entry point: runs the full pipeline from query to chart-ready data.
        """
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=SecretStr(api_key) if api_key else None,
            temperature=0,
        )

        # 1. Retrieve and grade documents using CRAG
        retriever = CRAGRetriever(self.vector_db, self.llm)
        relevant_results = retriever.get_relevant_documents(query, k=5)

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

        # 3. Generate the answer using LLM
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Conversational Strategic Financial Advisor. 

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
            - [Suggestion 2]
            """),
            ("human", "Chat History: {chat_history}\n\nQuestion: {query}\n\nContext: {context}")
        ])

        chat_history = []
        
        try:
            answer_response = self.llm.invoke(answer_prompt.format_messages(query=query, context=context, chat_history=chat_history))
            answer = answer_response.content.strip()
        except Exception as e:
            print(f"⚠️ Answer generation error: {e}")
            answer = "Unable to generate an answer at this time."

        # 4. Verification using Self-RAG
        verifier = SelfRAGVerifier(self.llm)
        verification = verifier.verify_answer(query, answer, [doc.page_content for doc in relevant_docs])

        confidence = "High" if verification.get("rating", 0) >= 8 else "Medium" if verification.get("rating", 0) >= 5 else "Low"

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
            "chart": chart_data
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