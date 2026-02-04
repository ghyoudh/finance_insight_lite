import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

current_dir = Path(__file__).parent.parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Verify GROQ_API_KEY is set
if not os.getenv("GROQ_API_KEY"):
    print("❌ ERROR: GROQ_API_KEY not found!")
    print(f"   Please add it to: {env_path}")
    sys.exit(1)

def clean_text(text):
    """clean text by removing punctuation and extra spaces, and converting to lowercase"""
    if not text: return ""

    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    return " ".join(text.split())

def calculate_rag_metrics(retrieved_context, generated_answer):
    """
    calculate (Precision, Recall, F1)
    """
    # convert both context and answer to sets of words for comparison
    context_words = set(clean_text(retrieved_context).split())
    answer_words = set(clean_text(generated_answer).split())

    # if either is empty, return 0 for all metrics
    if not answer_words or not context_words:
        return 0.0, 0.0, 0.0

    # words in common between retrieved context and generated answer
    hits = len(context_words.intersection(answer_words))

    # 1. Precision
    precision = hits / len(answer_words)
    
    # 2. Recall
    recall = hits / len(context_words)
    
    # 3. F1-Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return round(precision, 2), round(recall, 2), round(f1, 2)


if __name__ == "__main__":
    test_context = "Draw a pie chart of expense breakdown"
    test_answer = """To draw a pie chart of the expense breakdown, we need to analyze the provided data. Based on the information, the total depreciation expense for the year ended December 31, 2023, is 13,264 (in thousands). The breakdown of depreciation expense by class of assets is as follows: Land and land improvements (161), Buildings (510), Oil and gas properties (11), Plant, machinery and equipment (12,116), Depots, storage tanks and pipelines (338), and Fixtures, IT and office equipment (128).

        Suggestions
        Consider allocating a larger portion of the budget to Plant, machinery and equipment, as it accounts for the majority of the depreciation expense (approximately 91.5% of the total depreciation expense).
        Review the depreciation expense for Oil and gas properties, as it is relatively low (less than 1% of the total depreciation expense), and consider potential opportunities to optimize expenses in this area, potentially in Saudi Riyals (SAR)."""
    
    p, r, f1 = calculate_rag_metrics(test_context, test_answer)
    print(f"Test Precision: {p}, Recall: {r}, F1: {f1}")