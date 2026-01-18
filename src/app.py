import pathlib
from finance_insight_lite.modules.processor import pdf_to_markdown
from finance_insight_lite.modules.verctor_store import build_vector_db
from finance_insight_lite.modules.chat_engine import get_rag_chain

from dotenv import load_dotenv
import os
from pathlib import Path

# Get the project root directory (parent of src/)
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'

# Load the .env file
load_dotenv(dotenv_path=env_path)

# Debug: Check if the key is loaded
groq_key = os.getenv('GROQ_API_KEY')
if groq_key:
    print(f"✓ API Key loaded: {groq_key[:10]}...")
else:
    print("✗ API Key NOT loaded!")
    print(f"Looking for .env at: {env_path}")
    print(f".env exists: {env_path.exists()}")

if __name__ == "__main__":
    pdf_file = "data/rew/saudi-aramco-q3-2025-interim-report-english.pdf"
    content = pdf_to_markdown(pdf_file)
    # print the first 500 characters of the extracted content
    print("\nPart of the extracted content:")
    print(content[:500])

    print("-" * 30)
    verctor_db_path = pathlib.Path("./data/database")
    if not verctor_db_path.exists():
        verctor_db_path.mkdir(parents=True)
    vector_db = build_vector_db(content)
    print("Vector database has been built and persisted.")

    print("-" * 30)
    user_question = "What is the net income for Q3 2025?"
    print(f"Question: {user_question}")

    rag_chain = get_rag_chain(vector_db)
    response = rag_chain.invoke(user_question)
    source_docs = response.get('context')

    print("\nAI Response:")
    print(response)