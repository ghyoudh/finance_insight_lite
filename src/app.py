import pathlib
import re
from finance_insight_lite.modules.processor import pdf_to_documents
from finance_insight_lite.modules.verctor_store import build_vector_db
from finance_insight_lite.modules.rag_agent import create_advanced_rag_agent


from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables from .env file
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
    content = pdf_to_documents(pdf_file)
    # print the first 300 characters of the extracted content
    print("\nPart of the extracted content:")
    print(content[:300])

    print("-" * 30)
    # Build or load vector database
    verctor_db_path = pathlib.Path("./data/database")
    if not verctor_db_path.exists():
        verctor_db_path.mkdir(parents=True)
    vector_db = build_vector_db(content)
    print("Vector database has been built and persisted.")

    print("-" * 30)
    # Create agent
    # use_self_rag=True for highest accuracy (but slower)
    # use_self_rag=False for speed with good accuracy
    agent = create_advanced_rag_agent(
        vector_db=vector_db,
        use_self_rag=True  # Change to False for faster response
    )
    
    test_questions = [
        "What is the net income for Q3 2025?",
        "What is the free cash flow?",
        "What is the gearing ratio?",
        "How much was the dividend declared?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'#'*60}")
        print(f"Question {i}: {question}")
        print(f"{'#'*60}")
        
        # Process the question
        result = agent.process_query(question)
        
        # Display results
        print(f"\n📊 Result:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Answer: {result['answer']}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📄 Source Pages: {', '.join(map(str, result['source_pages']))}")
        print(f"🎯 Confidence Level: {result['confidence']}")
        print(f"📈 Relevant Documents Count: {result['relevant_docs_count']}")
        
        if result['verification']:
            print(f"\n🔍 Self-Verification Result:")
            print(f"   {result['verification']['verification'][:200]}...")
            print(f"   ✅ Passed" if result['verification']['passed'] else "   ⚠️ Failed")
        
        print()
    
    print("\n" + "="*60)
    print("✅ Testing completed successfully!")
    print("="*60)