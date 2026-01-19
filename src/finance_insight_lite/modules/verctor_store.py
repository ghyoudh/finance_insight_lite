from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_db(documents, db_path="./database"):
    """
    Build vector database from documents with preserved metadata using FAISS
    """
    print(f"Building vector DB from {len(documents)} documents")

    # Split documents into chunks while preserving metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)

    print(f"✓ Created {len(chunks)} chunks from {len(documents)} pages")

    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vector database (better metadata handling than Chroma)
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Save to disk
    vector_db.save_local(db_path)

    print(f"✓ Vector database saved to: {db_path}")

    return vector_db