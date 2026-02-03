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
            f"[Page {doc.metadata.get('page', '?')} | Sheet: {doc.metadata.get('sheet_name', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        ])
        
        # Use LLM to extract structured data
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial data extraction specialist.
            
            Extract structured numerical data from the provided text.
            Return ONLY a valid JSON array of objects. No explanations, no markdown, no ```json``` blocks.
            
            Rules:
            - Each object must have a "label" key and a "value" key
            - "value" must be a number (remove commas, currency symbols, and units)
            - If you find quarterly data, use labels like "Q1 2025", "Q2 2025"
            - If you find yearly data, use labels like "2024", "2025"
            - NEVER include empty values or null values
            - Extract as many data points as possible
            - Only extract data relevant to the query
            
            Example output:
            [{"label": "Q1 2025", "value": 25000}, {"label": "Q2 2025", "value": 28000}]
            """),
            ("human", f"Query: {query}\n\nText:\n{combined_text}\n\nExtracted JSON:")
        ])
        
        try:
            response = self.llm.invoke(extraction_prompt)
            raw = response.content.strip()
            
            # Clean up the response - remove markdown code blocks if present
            raw = re.sub(r'```(?:json)?\s*', '', raw)
            raw = raw.replace('```', '').strip()
            
            # Parse JSON
            data = json.loads(raw)
            
            if isinstance(data, list) and len(data) > 0:
                # Filter out empty or invalid entries
                valid_data = []
                for item in data:
                    if isinstance(item, dict) and 'label' in item and 'value' in item:
                        label = str(item['label']).strip()
                        try:
                            value = float(str(item['value']).strip())
                            if label and value >= 0:  # Only keep valid entries
                                valid_data.append({'label': label, 'value': value})
                        except (ValueError, TypeError):
                            continue
                
                if valid_data:
                    df = pd.DataFrame(valid_data)
                    print(f"✅ Extracted {len(df)} valid data points using LLM")
                    return df
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            print(f"   Raw response: {raw[:200]}")
        except Exception as e:
            print(f"⚠️ LLM extraction error: {e}")
        
        # Fallback: Manual regex parsing
        print("📌 Falling back to regex extraction...")
        return self._regex_extract(combined_text)
    
    def _regex_extract(self, text: str) -> pd.DataFrame:
        """Fallback regex-based extraction for common financial patterns"""
        data = []
        
        # Pattern 1: "Q1 2025 ... $25.0 billion" or "Q1 2025: 25,000"
        quarterly_pattern = r'(Q[1-4]\s*\d{4})[^\d]*?([\d,]+\.?\d*)\s*(?:billion|million|bn|mn|مليار|مليون)?'
        for match in re.finditer(quarterly_pattern, text, re.IGNORECASE):
            label = match.group(1).strip()
            try:
                value = float(match.group(2).replace(',', ''))
                if label and value >= 0:
                    data.append({"label": label, "value": value})
            except (ValueError, AttributeError):
                continue
        
        # Pattern 2: "Label: $value" or "Label | value"
        label_value_pattern = r'([A-Za-z\s]+(?:income|revenue|profit|cost|expense|cash|flow|ratio|margin|dividend|EBITDA|CapEx))[:\|]\s*\$?([\d,]+\.?\d*)'
        for match in re.finditer(label_value_pattern, text, re.IGNORECASE):
            label = match.group(1).strip()
            try:
                value = float(match.group(2).replace(',', ''))
                if label and value >= 0:
                    data.append({"label": label, "value": value})
            except (ValueError, AttributeError):
                continue
        
        # Pattern 3: Year-based "2023 ... value"
        year_pattern = r'(20\d{2})[^\d]*?([\d,]+\.?\d*)\s*(?:billion|million|bn|mn)?'
        for match in re.finditer(year_pattern, text):
            label = match.group(1)
            try:
                value = float(match.group(2).replace(',', ''))
                if value > 0:
                    data.append({"label": label, "value": value})
            except (ValueError, AttributeError):
                continue
        
        if data:
            df = pd.DataFrame(data)
            # Remove duplicates
            df = df.drop_duplicates(subset=['label'])
            # Ensure value is numeric and drop any NaN
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            print(f"✅ Regex extracted {len(df)} valid data points")
            return df if len(df) > 0 else pd.DataFrame()
        
        return pd.DataFrame()

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
        fig.update_traces(textposition='inside', textinfo='percent+label')
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
##
class FinancialRAGAgent:
    """Extract financial data from documents for visualization"""
    
    def __init__(self, vector_db):
        self.vector_db = vector_db
    
    def extract_data_from_query(self, query: str, k: int = 8) -> pd.DataFrame:
        """
        Extract numerical data using LLM-assisted parsing
        """
        api_key = os.getenv("GROQ_API_KEY")
        
        # Use faster model configuration
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=SecretStr(api_key) if api_key else None,
            temperature=0,
        )

        docs = self.vector_db.similarity_search(query, k=k)
        
        if not docs:
            return pd.DataFrame()
        
        combined_text = "\n\n".join([
            f"[Page {doc.metadata.get('page', '?')} | Sheet: {doc.metadata.get('sheet_name', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        ])
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial data extraction specialist.
            
            Extract structured numerical data from the provided text.
            Return ONLY a valid JSON array of objects. No explanations, no markdown, no ```json``` blocks.
            
            STRICT Rules:
            - Each object must have a "label" key and a "value" key
            - "label" must NEVER be empty or null
            - "value" must ALWAYS be a valid number (no empty strings, no null, no text)
            - Remove commas, currency symbols (﷼, $), and units (billion, million) from values
            - If you find quarterly data, use labels like "Q1 2025", "Q2 2025"
            - If you find yearly data, use labels like "2024", "2025"
            - Only include entries where BOTH label and value are valid
            - Extract as many data points as possible
            - Keep your answer to 10-12 sentences maximum
            
            WRONG:
            [{"label": "", "value": ""}, {"label": "Revenue", "value": null}]
            
            CORRECT:
            [{"label": "Q1 2025", "value": 25000}, {"label": "Q2 2025", "value": 28000}]
            """),
            ("human", f"Query: {query}\n\nText:\n{combined_text}\n\nExtracted JSON:")
        ])
        
        try:
            response = self.llm.invoke(extraction_prompt)
            raw = response.content.strip()
            
            # Clean up markdown code blocks
            raw = re.sub(r'```(?:json)?\s*', '', raw)
            raw = raw.replace('```', '').strip()
            
            # Sometimes LLM adds text before/after the JSON array
            # Extract only the JSON array part
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                raw = json_match.group(0)
            
            data = json.loads(raw)
            
            if isinstance(data, list) and len(data) > 0:
                df = self._clean_dataframe(data)
                if not df.empty:
                    print(f"✅ Extracted {len(df)} valid data points using LLM")
                    return df
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            print(f"   Raw response: {raw[:200]}")
        except Exception as e:
            print(f"⚠️ LLM extraction error: {e}")
        
        # Fallback
        print("📌 Falling back to regex extraction...")
        return self._regex_extract(combined_text)
    
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
    
    def _regex_extract(self, text: str) -> pd.DataFrame:
        """Fallback regex-based extraction"""
        data = []
        
        # Pattern 1: Quarterly "Q1 2025 ... number"
        quarterly_pattern = r'(Q[1-4]\s*\d{4})[^\d]*?([\d,]+\.?\d*)\s*(?:billion|million|bn|mn|مليار|مليون)?'
        for match in re.finditer(quarterly_pattern, text, re.IGNORECASE):
            label = match.group(1).strip()
            value_str = match.group(2).replace(',', '').strip()
            if not value_str:
                continue
            try:
                value = float(value_str)
                data.append({"label": label, "value": value})
            except ValueError:
                continue
        
        # Pattern 2: "Financial Label: value"
        label_value_pattern = r'([A-Za-z\s]+(?:income|revenue|profit|cost|expense|cash|flow|ratio|margin|dividend|EBITDA|CapEx))[:\|]\s*\$?([\d,]+\.?\d*)'
        for match in re.finditer(label_value_pattern, text, re.IGNORECASE):
            label = match.group(1).strip()
            value_str = match.group(2).replace(',', '').strip()
            if not value_str or not label:
                continue
            try:
                value = float(value_str)
                data.append({"label": label, "value": value})
            except ValueError:
                continue
        
        # Pattern 3: Year-based "2023 ... value"
        year_pattern = r'(20\d{2})[^\d]*?([\d,]+\.?\d*)\s*(?:billion|million|bn|mn)?'
        for match in re.finditer(year_pattern, text):
            label = match.group(1).strip()
            value_str = match.group(2).replace(',', '').strip()
            if not value_str:
                continue
            try:
                value = float(value_str)
                if value > 0:
                    data.append({"label": label, "value": value})
            except ValueError:
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset=['label'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value', 'label'])
        # Remove rows where label is empty string
        df = df[df['label'].str.strip() != '']
        
        print(f"✅ Regex extracted {len(df)} valid data points")
        return df if len(df) > 0 else pd.DataFrame()