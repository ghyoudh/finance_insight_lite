# Finance Insight Lite 📊💰

An intelligent financial document analysis system powered by **Retrieval-Augmented Generation (RAG)** and advanced language models. Instantly extract insights from financial reports, earnings statements, and Excel data with AI-driven Q&A and interactive visualizations.

## ✨ Key Features

### 🚀 Lightning-Fast Document Processing
- **Parallel PDF/Excel Processing**: Process multiple files simultaneously with ThreadPoolExecutor
- **Smart Caching System**: Instant reload using file-hash based caching
- **Optimized Chunking**: Intelligent document segmentation for better retrieval
- **Multi-format Support**: PDF and Excel spreadsheets (.xlsx, .xls)

### 🤖 Advanced RAG Engine
- **Corrective RAG (CRAG)**: Batch document grading for high-quality retrieval
- **Self-RAG Verification**: Optional fact-checking and answer validation
- **LLM-Assisted Data Extraction**: Intelligent extraction of financial metrics
- **Semantic Vector Search**: Chroma + FAISS for semantic similarity matching

### 📊 Interactive Data Visualization
- **Multiple Chart Types**: Bar, Line, Pie, Scatter, and Area charts
- **Auto Data Extraction**: LLM-powered extraction of numerical data from documents
- **Fallback to Demo Data**: Demo data displays if no data found
- **Data Table Export**: View extracted data in tabular format

### 💬 Modern Web Interface
- **Streamlit UI**: Clean, responsive web interface
- **Real-time Chat**: Conversation history with context awareness
- **Source Attribution**: See exact page references for all answers
- **Configuration Panel**: Fine-tune RAG parameters on-the-fly

### ⚙️ Customizable Settings
- **Self-RAG Toggle**: Balance between accuracy and speed
- **Relevance Threshold**: Control answer precision
- **Document Count**: Adjust number of context documents
- **Chart Type Selection**: Choose default visualization type
- **Cache Management**: View and clear cache with one click

## 🎯 What It Can Do

### Financial Analysis
- Extract key metrics (revenue, profit, cash flow, ratios)
- Answer questions about financial performance
- Provide trend analysis and comparisons
- Calculate financial ratios and metrics

### Document Intelligence
- Semantic search across multiple documents
- Multi-document cross-referencing
- Context-aware question answering
- Source page tracking

### Data Visualization
- Create charts from financial data
- Visualize trends and patterns
- Compare metrics side-by-side
- Export data for further analysis

## 🚀 Quick Start

### Prerequisites
- **Python**: 3.11 or higher
- **Groq API Key**: Get free at https://console.groq.com
- **RAM**: Minimum 4GB (8GB recommended)

### 1. Clone Repository
```bash
cd Finance_Insight_Lite
```

### 2. Create Virtual Environment
```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Create `.env` file in project root:
```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

### 5. Run Application
```bash
# Web UI (Recommended)
streamlit run src/ui.py

# Opens at http://localhost:8501
```
---



---

## 📁 Project Structure

```
Finance_Insight_Lite/
├── data/                    # Document storage and processing
│   ├── cache/
│   ├── database/
│   ├── processed/
│   ├── rew/
│   ├── uploaded/
│   └── vector_db/
├── database/                # Local vector store indices
├── images/                  # UI assets
│   ├── chatbots_icon.png
│   ├── logo.png
│   └── user_icon.png
├── node_modules/            # Node.js dependencies
├── src/
│   ├── finance_insight_lite/
│   │   ├── modules/
│   │   │   ├── processor.py
│   │   │   ├── rag_agent.py        # RAG agent implementation ⭐
│   │   │   └── verctor_store.py
│   │   └── __init__.py
│   ├── app.py               # CLI entry point
│   └── ui.py                # Streamlit web interface ⭐
├── .env                     # Environment variables
├── .gitignore               # Git exclusion rules
├── main.py                  # API entry point (FastAPI service) ⭐
├── package-lock.json
├── package.json
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock                  # UV dependency lockfile
```

## 🔧 Core Modules

### `rag_agent.py` (Main RAG Engine)
The heart of the system with four main components:

#### 1. **CRAGRetriever** - Corrective RAG
```python
# Batch-grades documents for relevance
retriever = CRAGRetriever(vector_db, llm)
relevant_docs = retriever.get_relevant_documents(question, k=5)
```
- Retrieves documents semantically similar to query
- Batch grades all documents for relevance
- Filters to only high-confidence matches
- Fallback mechanism if no documents match

#### 2. **SelfRAGVerifier** - Answer Verification
```python
# Validates answers against sources
verifier = SelfRAGVerifier(llm)
verification = verifier.verify_answer(question, answer, sources)
```
- Rates answer accuracy (1-10 scale)
- Checks source support
- Optional (can be disabled for speed)

#### 3. **FinancialDataExtractor** - Data Extraction
```python
# Extracts numerical data from documents
extractor = FinancialDataExtractor(vector_db, llm)
df = extractor.extract_data_from_query("net income")
```
- LLM-assisted extraction with JSON parsing
- Regex fallback for robust parsing
- Automatic demo data if no data found
- Data validation and cleaning

#### 4. **ChartGenerator** - Visualizations
```python
# Creates interactive Plotly charts
generator = ChartGenerator()
fig = generator.create_bar_chart(df, 'label', 'value', 'Revenue by Quarter')
```
- Bar, Line, Pie, Scatter, Area charts
- Handles categorical and numeric axes
- Financial styling with color palettes
- Responsive layout

### `processor.py` - Document Processing
```python
# All-in-one processing with caching
result = load_documents_fastest(
    file_path="report.pdf",
    use_cache=True,
    max_workers=2
)
documents = result['documents']
file_type = result['file_type']
```

**Features**:
- Parallel PDF/Excel processing
- Automatic file-hash caching
- Smart chunking strategies
- Metadata preservation

### `verctor_store.py` - Vector Database
```python
# Build and manage vector embeddings
db = build_vector_db(documents, db_path="./database")
results = db.similarity_search(query, k=5)
```

**Features**:
- Chroma + FAISS integration
- HuggingFace embeddings
- Persistent storage
- Similarity search

## 🎨 User Interface Features

### Sidebar Controls
- **Document Upload**: Support for PDF and Excel files
- **Processing**: One-click document processing with progress bar
- **Settings**: Customizable RAG parameters
- **Cache Management**: View cache size and clear cache

### Main Chat Interface
- **Message History**: Full conversation context
- **Source Attribution**: Page references for answers
- **Confidence Scores**: High/Medium/Low confidence indicators
- **Metadata Display**: Documents used, verification scores
- **Charts & Tables**: Integrated visualizations with data export

### Sample Questions
Quick-start buttons for common queries:
- "What is the net income?"
- "What is the free cash flow?"
- "What is the gearing ratio?"
- "How much was the dividend declared?"
- "Draw a pie chart of expense breakdown"
- "Visualize the profit margins"

## 🔑 Configuration Guide

### Environment Variables
```env
# Required
GROQ_API_KEY=your_key_here

# Optional
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
```

### RAG Settings (In-App)
| Setting | Default | Range | Effect |
|---------|---------|-------|--------|
| **Self-RAG** | Enabled | On/Off | Accuracy vs Speed |
| **Relevance Threshold** | 0.6 | 0.0-1.0 | Answer precision |
| **Doc Retrieval** | 3 | 3-10 | Context coverage |
| **Chart Type** | bar | - | Default visualization |
| **Show Data Table** | On | On/Off | Data export option |

## 📊 Visualization Examples

### Supported Chart Types

**Bar Chart**
```
"Draw a bar chart of quarterly revenue"
→ Shows revenue by quarter with numeric comparison
```

**Line Chart**
```
"Plot profit margin trends"
→ Shows trends over time with markers
```

**Pie Chart**
```
"Create a pie chart of expense breakdown"
→ Shows proportional distribution
```

**Scatter Chart**
```
"Visualize correlation between metrics"
→ Shows relationships with optional trendline
```

**Area Chart**
```
"Display cumulative revenue"
→ Shows stacked area representation

## 🔐 API Keys

### Groq API
1. Visit https://console.groq.com
2. Sign up for free account
3. Generate API key
4. Add to `.env` file

**Why Groq?**
- ⚡ Ultra-fast inference (100+ tokens/sec)
- 💰 Generous free tier
- 🤖 Latest LLaMA models
- 📈 High accuracy for financial analysis

## 📈 Performance Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| PDF Processing | 2-5s | 50-100 pages, cached after 1st run |
| Excel Processing | 1-3s | Up to 10,000 rows |
| Query Response | 3-8s | With retrieval + verification |
| Chart Generation | <1s | From extracted data |
| Document Caching | <100ms | Instant reload |

## 🐛 Troubleshooting

### Chart Not Displaying
- Ensure document contains numeric data
- Try a different chart type
- Check data extraction in logs

### "No relevant documents found"
- Increase relevance threshold lower
- Ask more specific questions
- Ensure documents are uploaded

### Slow Processing
- Disable Self-RAG for faster responses
- Reduce document retrieval count
- Clear cache and re-upload

### API Errors
- Verify GROQ_API_KEY is correct
- Check internet connection
- Review API quota at console.groq.com

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Building Distribution
```bash
pip install build
python -m build
```

### Code Quality
```bash
pip install pylance black flake8
black src/
flake8 src/
```

## 📝 Usage Examples

### Example 1: Extract & Visualize Revenue
```
User: "Draw a bar chart for the net income"
→ System: Extracts net income from documents
→ Creates interactive bar chart
→ Shows data table with values
```

### Example 2: Multi-Document Analysis
```
User: "What was the total revenue across all quarters?"
→ System: Retrieves relevant sections from all documents
→ Aggregates quarterly data
→ Provides answer with page references
```

### Example 3: Financial Ratio Analysis
```
User: "What is the gearing ratio?"
→ System: Finds financial statements
→ Calculates ratio from relevant metrics
→ Provides context and interpretation
```

## 📚 Resources

- **LangChain Docs**: https://docs.langchain.com
- **Groq Docs**: https://console.groq.com/docs
- **Streamlit Docs**: https://docs.streamlit.io
- **Plotly Docs**: https://plotly.com/python

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💼 About

**Finance Insight Lite** is a lightweight yet powerful tool for financial document analysis. Built with modern AI/ML technologies, it makes financial intelligence accessible to everyone.

**Current Version**: 0.2.0  
**Last Updated**: February 3, 2026  
**Status**: Active Development ✅

---

## ⭐ Quick Tips

1. **Upload multiple files** at once for comprehensive analysis
2. **Use specific keywords** in questions for better results
3. **Check source pages** to validate answers
4. **Experiment with chart types** to find best visualization
5. **Clear cache periodically** to free up disk space
6. **Adjust relevance threshold** if getting irrelevant results

---

**Questions or Issues?** Create an issue on GitHub or contact support.

Happy analyzing! 📈🚀
