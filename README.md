# Finance Insight Lite 📊💰

An intelligent financial document analysis system powered by **Retrieval-Augmented Generation (RAG)** and advanced language models. Instantly extract insights from financial reports, earnings statements, and Excel data with AI-driven Q&A.

## Features ✨

### 🚀 High-Performance Document Processing
- **Parallel PDF Processing**: 3-5x faster extraction using ThreadPoolExecutor
- **Intelligent Caching**: Instant reload of previously processed documents via file hashing
- **Optimized Excel Handling**: Auto-chunking for large spreadsheets with smart metadata
- **Fast PyMuPDF Integration**: Optimized text extraction from PDFs

### 🤖 Advanced RAG Capabilities
- **Self-Correcting RAG**: Evaluates retrieval quality and adjusts search strategy
- **Multi-Document Analysis**: Process multiple financial documents simultaneously
- **Relevance Filtering**: Threshold-based filtering for high-quality responses
- **Vector Database**: Semantic search using Chroma with FAISS indexing

### 💬 Interactive Chat Interface
- **Streamlit Web UI**: User-friendly interface for document upload and querying
- **Chat History**: Maintain conversation context across sessions
- **Source Attribution**: View exact sources for all responses
- **Real-time Processing**: Progress tracking with visual feedback

### ⚙️ Configuration & Control
- **Self-RAG Toggle**: Switch between accuracy and speed
- **Relevance Threshold Control**: Fine-tune answer precision
- **Document Retrieval Settings**: Adjust number of context documents
- **Cache Management**: Clear cache with size display

## Quick Start 🚀

### Prerequisites
- Python 3.11 or higher
- GROQ API Key (get one at https://console.groq.com)

### Installation

1. **Clone and navigate to the project:**
```bash
cd Finance_Insight_Lite
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate   # Linux/macOS
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_key_here
```

### Running the Application

**Option 1: Web Interface (Recommended)**
```bash
streamlit run src/ui.py
```
Then open your browser to `http://localhost:8501`

**Option 2: Python Script**
```bash
python src/app.py
```

## Project Structure 📁

```
Finance_Insight_Lite/
├── src/
│   ├── app.py                          # CLI/script entry point
│   ├── ui.py                           # Streamlit web interface
│   ├── database/                       # Vector database files
│   │   ├── chroma.sqlite3
│   │   └── index.faiss
│   ├── data/
│   │   ├── uploaded/                   # User uploaded files
│   │   ├── cache/                      # Cached processed documents
│   │   └── processed/                  # Processed documents
│   ├── images/
│   │   └── logo.png                    # App logo
│   └── finance_insight_lite/
│       ├── __init__.py
│       └── modules/
│           ├── processor.py            # PDF/Excel processing with caching
│           ├── verctor_store.py        # Vector DB management
│           └── rag_agent.py            # RAG agent implementation
├── database/                           # Global vector store
├── data/                               # Data cache and uploads
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Project configuration
├── .env                                # Environment variables (create this)
└── README.md                           # This file
```

## Core Modules 🔧

### `processor.py`
Handles all document processing with optimization layers:
- **PDF Processing**: `pdf_to_documents_cached()`, `pdf_to_documents_parallel()`, `pdf_to_documents_fast()`
- **Excel Processing**: `excel_to_documents_optimized()`
- **Main API**: `load_documents_fastest()` - unified interface with all optimizations
- **Caching**: Automatic file hash-based caching system
- **Cache Management**: `clear_cache()` to free up disk space

### `verctor_store.py`
Vector database management:
- Chroma + FAISS integration for semantic search
- Document storage and retrieval
- Vector embeddings using HuggingFace models

### `rag_agent.py`
RAG agent implementation:
- LangChain integration
- Self-RAG for answer validation
- Query rewriting and context injection
- Groq API integration for fast inference

## Performance Optimizations ⚡

| Feature | Improvement |
|---------|-------------|
| Caching | 5-10x faster on repeat loads |
| Parallel Processing | 3-5x faster for large PDFs |
| Optimized Pandas | 2-3x faster Excel processing |
| Vector Search | Semantic matching vs. keyword search |
| Self-RAG | Automatic answer validation & refinement |

## API Keys & Configuration 🔐

Required for full functionality:

| Service | Variable | Where to Get |
|---------|----------|-------------|
| Groq | `GROQ_API_KEY` | https://console.groq.com |

## Usage Examples 📚

### Web Interface
1. Upload PDF or Excel files via sidebar
2. Click "Process All Documents"
3. Ask questions in the chat box
4. View answers with source attribution

---

**Last Updated**: 27 January 2026  
**Version**: 0.1.0