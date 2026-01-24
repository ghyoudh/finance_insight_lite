import fitz  # PyMuPDF
import pandas as pd
import os
from langchain_core.documents import Document

def pdf_to_documents(pdf_path):
    """
    Load PDF and return documents with page metadata preserved
    """
    print(f"Loading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    documents = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        # Create a Document with page metadata
        document = Document(
            page_content=text,
            metadata={
                "source": os.path.basename(pdf_path),
                "page": page_num + 1  # Pages start from 1, not 0
            }
        )
        documents.append(document)
    print(f"✓ Loaded {len(documents)} pages from PDF")
    return documents

def excel_to_documents(excel_path, sheet_name=None):
    """
    Load Excel file and return documents with sheet metadata preserved
    
    Args:
        excel_path: Path to the Excel file
        sheet_name: Specific sheet name to load, or None to load all sheets
    """
    print(f"Loading Excel: {excel_path}")
    documents = []
    
    # Read Excel file
    if sheet_name:
        # Load specific sheet
        sheets = {sheet_name: pd.read_excel(excel_path, sheet_name=sheet_name)}
    else:
        # Load all sheets
        sheets = pd.read_excel(excel_path, sheet_name=None)
    
    # Process each sheet
    for sheet_name, df in sheets.items():
        # Convert DataFrame to string representation
        text_content = f"Sheet: {sheet_name}\n\n"
        text_content += df.to_string(index=False)
        
        # Create a Document with sheet metadata
        document = Document(
            page_content=text_content,
            metadata={
                "source": os.path.basename(excel_path),
                "sheet_name": sheet_name,
                "rows": len(df),
                "columns": len(df.columns)
            }
        )
        documents.append(document)
    
    print(f"✓ Loaded {len(documents)} sheets from Excel")
    return documents

def load_documents(file_path, **kwargs):
    """
    Universal document loader that handles both PDF and Excel files
    
    Args:
        file_path: Path to the file
        **kwargs: Additional arguments (e.g., sheet_name for Excel)
    
    Returns:
        dict: Contains 'documents' list, 'relevant_docs_count', and 'file_type'
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        docs = pdf_to_documents(file_path)
        file_type = 'PDF'
    elif file_extension in ['.xlsx', '.xls']:
        docs = excel_to_documents(file_path, **kwargs)
        file_type = 'Excel'
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    return {
        'documents': docs,
        'relevant_docs_count': len(docs),
        'file_type': file_type,
        'source': os.path.basename(file_path)
    }