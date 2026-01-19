import fitz  # PyMuPDF
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