import hashlib
import json
import pickle
from functools import lru_cache
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _hash_file(file_path):
    digest = hashlib.md5()
    with open(file_path, "rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_key(documents, source_paths):
    """Return a stable cache key for the corpus and index configuration."""
    if source_paths:
        source_fingerprints = [_hash_file(path) for path in source_paths]
    else:
        source_fingerprints = [
            hashlib.md5(
                json.dumps(
                    {"page_content": document.page_content, "metadata": document.metadata},
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            for document in documents
        ]
    configuration = {
        "sources": source_fingerprints,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
    }
    return hashlib.md5(json.dumps(configuration, sort_keys=True).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def get_embedding_model():
    """Create the embedding model once per Python process."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _load_or_create_chunks(documents, chunks_file):
    if chunks_file.exists():
        with open(chunks_file, "rb") as cache_file:
            chunks = pickle.load(cache_file)
        print(f"📦 Loaded {len(chunks)} chunks from cache")
        return chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    with open(chunks_file, "wb") as cache_file:
        pickle.dump(chunks, cache_file)
    print(f"✓ Created and cached {len(chunks)} chunks from {len(documents)} pages")
    return chunks


def build_vector_db(documents, db_path="./database", source_paths=None, cache_dir="data/vector_cache"):
  
    if not documents:
        raise ValueError("Cannot build a vector database without documents")
    cache_path = Path(cache_dir) / _cache_key(documents, source_paths)
    cache_path.mkdir(parents=True, exist_ok=True)
    index_file = cache_path / "index.faiss"
    metadata_file = cache_path / "index.pkl"
    hybrid_chunks_file = cache_path / "chunks_for_hybrid.pkl"
    embeddings = get_embedding_model()

    if index_file.exists() and metadata_file.exists():
        print(f"📦 Loading FAISS index from cache: {cache_path}")
        vector_db = FAISS.load_local(
            str(cache_path), embeddings, allow_dangerous_deserialization=True
        )
        if hybrid_chunks_file.exists():
            with open(hybrid_chunks_file, "rb") as f:
                chunks = pickle.load(f)
        else:
            chunks = _load_or_create_chunks(documents, cache_path / "chunks.pkl")
            with open(hybrid_chunks_file, "wb") as f:
                pickle.dump(chunks, f)
        return vector_db, chunks

    print(f"Building vector DB from {len(documents)} documents")
    chunks = _load_or_create_chunks(documents, cache_path / "chunks.pkl")
    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_db.save_local(str(cache_path))

    with open(hybrid_chunks_file, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✓ Cached FAISS index at: {cache_path}")
    return vector_db, chunks
