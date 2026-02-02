"""
Financial Data Visualization Tool
Enables the RAG agent to create charts from financial data
"""

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json


# ============================================================================
# TOOL INPUT SCHEMA
# ============================================================================

class VisualizationInput(BaseModel):
    """Input schema for the visualization tool"""
    chart_type: Literal["line", "bar", "pie", "scatter", "area"] = Field(
        description="Type of chart to create: 'line' for trends, 'bar' for comparisons, 'pie' for proportions, 'scatter' for correlations, 'area' for cumulative data"
    )
    data_query: str = Field(
        description="Description of what data to extract (e.g., 'quarterly revenue', 'net income by year', 'expense breakdown')"
    )
    x_axis: Optional[str] = Field(
        default=None,
        description="Column name for X-axis (e.g., 'Quarter', 'Year', 'Month')"
    )
    y_axis: Optional[str] = Field(
        default=None,
        description="Column name for Y-axis (e.g., 'Revenue', 'Net Income', 'Profit')"
    )
    title: Optional[str] = Field(
        default=None,
        description="Chart title"
    )


# ============================================================================
# DATA EXTRACTOR
# ============================================================================

class FinancialDataExtractor:
    """Extract financial data from documents"""
    
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
        # Search for relevant documents
        docs = self.vector_db.similarity_search(query, k=k)
        
        # Extract tables and numerical data
        all_data = []
        
        for doc in docs:
            # Check if document is from Excel (has structured data)
            if doc.metadata.get('sheet_name'):
                # Parse Excel content
                data = self._parse_excel_content(doc.page_content)
                if data:
                    all_data.extend(data)
            else:
                # Parse PDF text for tables
                data = self._parse_pdf_tables(doc.page_content)
                if data:
                    all_data.extend(data)
        
        if not all_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        return df
    
    def _parse_excel_content(self, content: str) -> List[Dict]:
        """Parse Excel sheet content into structured data"""
        lines = content.split('\n')
        data = []
        
        # Simple parser - looks for table-like structure
        headers = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to split by whitespace (Excel tables are usually aligned)
            parts = line.split()
            
            if headers is None and len(parts) > 1:
                # First line with multiple columns is likely headers
                headers = parts
            elif headers and len(parts) == len(headers):
                # Data row
                row_dict = dict(zip(headers, parts))
                data.append(row_dict)
        
        return data
    
    def _parse_pdf_tables(self, content: str) -> List[Dict]:
        """Parse tables from PDF text content"""
        # Simple parser for common financial table patterns
        lines = content.split('\n')
        data = []
        
        # Look for patterns like "Q1 2025: $100M" or "Revenue | 1000"
        for line in lines:
            # Pattern 1: "Label: Value"
            if ':' in line and any(char.isdigit() for char in line):
                parts = line.split(':')
                if len(parts) == 2:
                    label = parts[0].strip()
                    value = parts[1].strip()
                    data.append({"label": label, "value": value})
            
            # Pattern 2: "Label | Value"
            elif '|' in line and any(char.isdigit() for char in line):
                parts = line.split('|')
                if len(parts) >= 2:
                    label = parts[0].strip()
                    value = parts[1].strip()
                    data.append({"label": label, "value": value})
        
        return data


# ============================================================================
# CHART GENERATOR
# ============================================================================

class ChartGenerator:
    """Generate Plotly charts from data"""
    
    @staticmethod
    def create_line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """Create a line chart"""
        fig = px.line(
            df,
            x=x,
            y=y,
            title=title,
            markers=True,
            template="plotly_white"
        )
        
        fig.update_layout(
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """Create a bar chart"""
        fig = px.bar(
            df,
            x=x,
            y=y,
            title=title,
            template="plotly_white"
        )
        
        fig.update_layout(
            showlegend=True,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_pie_chart(df: pd.DataFrame, names: str, values: str, title: str) -> go.Figure:
        """Create a pie chart"""
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
    
    @staticmethod
    def create_scatter_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """Create a scatter chart"""
        fig = px.scatter(
            df,
            x=x,
            y=y,
            title=title,
            template="plotly_white",
            trendline="ols"
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    @staticmethod
    def create_area_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """Create an area chart"""
        fig = px.area(
            df,
            x=x,
            y=y,
            title=title,
            template="plotly_white"
        )
        
        fig.update_layout(height=500)
        
        return fig


# ============================================================================
# VISUALIZATION TOOL
# ============================================================================

class VisualizeFinancialDataTool(BaseTool):
    """Tool for creating financial data visualizations"""
    
    name: str = "visualize_financial_data"
    description: str = """
    Use this tool when the user asks to visualize, plot, chart, or draw financial data.
    
    Examples of when to use:
    - "Show me a chart of quarterly revenue"
    - "Plot the net income trend"
    - "Draw a pie chart of expenses"
    - "Visualize the growth over time"
    
    This tool will:
    1. Extract the relevant financial data
    2. Create the appropriate chart type
    3. Return the chart for display
    """
    
    args_schema: type[BaseModel] = VisualizationInput
    vector_db: Any = None
    
    def _run(self, chart_type: str, data_query: str, x_axis: str = None, 
             y_axis: str = None, title: str = None) -> Dict[str, Any]:
        """Execute the visualization tool"""
        
        try:
            # 1. Extract data
            extractor = FinancialDataExtractor(self.vector_db)
            df = extractor.extract_data_from_query(data_query)
            
            if df.empty:
                return {
                    "success": False,
                    "error": "No data found for the query. Please try a different description.",
                    "chart": None
                }
            
            # 2. Auto-detect columns if not specified
            if not x_axis and len(df.columns) > 0:
                x_axis = df.columns[0]
            if not y_axis and len(df.columns) > 1:
                y_axis = df.columns[1]
            
            # 3. Generate title if not specified
            if not title:
                title = f"{chart_type.title()} Chart: {data_query}"
            
            # 4. Create chart
            generator = ChartGenerator()
            
            if chart_type == "line":
                fig = generator.create_line_chart(df, x_axis, y_axis, title)
            elif chart_type == "bar":
                fig = generator.create_bar_chart(df, x_axis, y_axis, title)
            elif chart_type == "pie":
                fig = generator.create_pie_chart(df, x_axis, y_axis, title)
            elif chart_type == "scatter":
                fig = generator.create_scatter_chart(df, x_axis, y_axis, title)
            elif chart_type == "area":
                fig = generator.create_area_chart(df, x_axis, y_axis, title)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported chart type: {chart_type}",
                    "chart": None
                }
            
            # 5. Convert to JSON for transmission
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


# ============================================================================
# AGENT INTEGRATION
# ============================================================================

def create_agent_with_visualization(vector_db, llm):
    """
    Create an agent with visualization capabilities
    
    Args:
        vector_db: Vector database with financial documents
        llm: Language model
    
    Returns:
        Agent with visualization tool
    """
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    # Create the visualization tool
    viz_tool = VisualizeFinancialDataTool(vector_db=vector_db)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analyst AI assistant with visualization capabilities.
        
        When users ask for visualizations (e.g., "draw", "plot", "chart", "show me"), use the visualize_financial_data tool.
        
        Guidelines:
        - For trends over time: use 'line' charts
        - For comparing categories: use 'bar' charts
        - For showing proportions: use 'pie' charts
        - For correlations: use 'scatter' charts
        - For cumulative data: use 'area' charts
        
        Always provide clear descriptions of what data you're visualizing."""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, [viz_tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[viz_tool], verbose=True)
    
    return agent_executor


# ============================================================================
# STANDALONE FUNCTION (for direct use without agent)
# ============================================================================

def visualize_data_direct(
    vector_db,
    query: str,
    chart_type: str = "bar",
    x_axis: str = None,
    y_axis: str = None,
    title: str = None
) -> Dict[str, Any]:
    """
    Direct visualization without using the agent
    
    Args:
        vector_db: Vector database
        query: Natural language query for data
        chart_type: Type of chart ('line', 'bar', 'pie', etc.)
        x_axis: X-axis column name
        y_axis: Y-axis column name
        title: Chart title
    
    Returns:
        Dictionary with chart and metadata
    """
    tool = VisualizeFinancialDataTool(vector_db=vector_db)
    return tool._run(chart_type, query, x_axis, y_axis, title)