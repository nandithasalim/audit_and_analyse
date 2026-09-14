"""
Run this first, before building anything else, to confirm your OpenAI key
and the web_search tool actually work end to end.

    python scripts/smoke_test.py

Expect: an answer with real citations, and a cost line that isn't zero.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import STRONG_MODEL
from src.llm import call_llm

if __name__ == "__main__":
    result = call_llm(
        "Which Indian jewellery retailer opened the most new stores in the last year? "
        "Answer in 2-3 sentences and cite your source.",
        question_id="smoke_test",
        stage="smoke_test",
        model=STRONG_MODEL,
        use_web_search=True,
    )

    print("\n--- ANSWER ---")
    print(result.text)

    print("\n--- CITATIONS ---")
    for c in result.citations:
        print(f"  - {c['title']}: {c['url']}")

    print("\n--- COST ---")
    print(f"  input tokens:  {result.input_tokens}")
    print(f"  output tokens: {result.output_tokens}")
    print(f"  search calls:  {result.search_calls}")
    print(f"  cost:          ${result.cost_usd:.4f}")
