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
from typing import List, Dict, Any, Literal

 
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
# 1. OPTIMIZED CRAG - Fast Retrieval with Batch Grading
# ============================================================================

class CRAGRetriever:
    """
    Corrective RAG مع تقييم دفعي عبر Structured Output (لا يوجد Regex)
    """
 
    # حد أدنى للثقة - أي تقييم بثقة أقل من هذا يُعامل كـ Irrelevant تلقائياً
    # (طبقة أمان إضافية: حتى لو صنّف النموذج المستند Relevant لكن بثقة ضعيفة،
    # نفضّل نحجبه بدل ما نجازف بعرض بيانات مالية غير دقيقة)
    MIN_CONFIDENCE_THRESHOLD = 0.6
 
    def __init__(self, vector_db, llm):
        self.vector_db = vector_db
        self.llm = llm
 
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
 
CRITICAL RULES:
- Do NOT write any preamble, explanation, or free text outside the schema.
- Return exactly one grade per document, in the same order given.
- If you are not reasonably confident a document is relevant, classify it as
  Irrelevant rather than guessing Moderately_Relevant. When uncertain, prefer
  the stricter (less relevant) label.
 
**Core Analytical Framework (for your reasoning, not for output format):**
1. Historical Performance Analysis - trends in revenue, net income, margins
2. Predictive Signal Detection - links between strategic decisions and outcomes
3. Strategic Context Evaluation - business logic behind financial changes
 
**Document Relevance Criteria:**
- Highly_Relevant: contains quantitative data or context directly answering the question
- Moderately_Relevant: partial/supporting context only
- Irrelevant: no real financial substance connected to the question"""),
            ("human", "Question: {question}\n\nDocuments:\n{document}\n\nGrade each document now via the schema.")
        ])
 
    def batch_grade_documents(self, question: str, documents: List[Any]) -> List[bool]:
        """
        تقييم دفعي بدون Regex - يعتمد كلياً على Structured Output
        """
        if not documents:
            return []
 
        docs_text = "\n\n".join([
            f"Document {i + 1} [Page {doc.metadata.get('page')}]:\n{doc.page_content[:1500]}"
            for i, doc in enumerate(documents)
        ])
 
        try:
            response: BatchGradeResponse = self.structured_llm.invoke(
                self.batch_grader_prompt.format(
                    question=question,
                    document=docs_text
                )
            )
 
            # نبني خريطة doc_index -> grade لضمان الترتيب الصحيح
            # حتى لو النموذج رجّع الترتيب بشكل مختلف
            grade_map = {g.doc_index: g for g in response.grades}
 
            relevance_results = []
            for i in range(1, len(documents) + 1):
                grade = grade_map.get(i)
 
                if grade is None:
                    # النموذج ما رجّع تقييم لهذا المستند -> فشل آمن: نحجبه
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
            # ✅ التصحيح الأهم: Fail-Closed وليس Fail-Open
            # القديم: return [True] * len(documents)  <- يمرر كل شيء وقت الخطأ (خطر)
            # الجديد: نحجب الكل وقت الخطأ، ونعتمد على fallback top-2 الموجود
            #         في get_relevant_documents لتفادي شاشة فاضية بالكامل
            print(f"❌ خطأ في التقييم الدفعي: {e} — سيتم حجب كل المستندات احترازياً")
            return [False] * len(documents)
 
    def get_relevant_documents(self, question: str, k: int = 5) -> List[Dict]:
        """
        استرجاع سريع مع تقييم دفعي آمن
 
        Args:
            question: السؤال المالي
            k: عدد المستندات المراد استرجاعها (افتراضي: 5)
 
        Returns:
            قائمة المستندات ذات الصلة
        """
        print(f"🔍 استرجاع {k} مستندات...")
 
        initial_docs = self.vector_db.similarity_search(question, k=k)
 
        if not initial_docs:
            return []
 
        relevance_flags = self.batch_grade_documents(question, initial_docs)
 
        relevant_results = [
            {"document": doc, "relevant": True}
            for doc, is_relevant in zip(initial_docs, relevance_flags)
            if is_relevant
        ]
 
        print(f"📊 عدد المستندات ذات الصلة: {len(relevant_results)}/{len(initial_docs)}")
 
        # هذا fallback مختلف عن fail-open: هنا فعلاً ما لقينا شيء relevant
        # (بعد تقييم حقيقي، مو بسبب خطأ تقني) فنرجع top-2 كحل احتياطي
        # حتى ما تطلع للمستخدم صفحة فاضية بالكامل
        if not relevant_results:
            print("⚠️ لم يُعثر على مستندات ذات صلة، استخدام أفضل 2 كحل احتياطي")
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

    def process_query(self, query: str , chat_history: list = None) -> dict:
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

        if chat_history is None: chat_history = []
        
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
