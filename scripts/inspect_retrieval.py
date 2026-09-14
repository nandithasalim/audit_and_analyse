"""
Debugging aid for the TF-IDF chunk-selection limitation documented in
src/retriever.py and DECISIONS.md: fetches a URL, chunks it, ranks every
chunk against a claim, and prints the full ranking -- not just the top-k
the auditor actually sees -- so you can find where the paragraph that
*actually* answers the claim landed.

Example (the worked case from DECISIONS.md):

    python scripts/inspect_retrieval.py \\
        --url "https://en.wikipedia.org/wiki/Tanishq" \\
        --claim "Tanishq is owned by Titan Company"

Look at which chunk (if any) contains the real answer, and what rank/score
it got vs. the chunks that were actually selected as top-5 evidence.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fetch import chunk_text, fetch_url
from src.retriever import rank_all_chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--claim", required=True)
    ap.add_argument("--top", type=int, default=None, help="only print the top N rows (default: all)")
    args = ap.parse_args()

    print(f"Fetching {args.url} ...")
    text = fetch_url(args.url)
    chunks = chunk_text(text, args.url)
    print(f"{len(chunks)} chunks extracted.\n")

    ranked = rank_all_chunks(args.claim, chunks)
    rows = ranked[: args.top] if args.top else ranked

    for rank, r in enumerate(rows, start=1):
        marker = " <-- top-5 (what the auditor would actually see)" if rank <= 5 else ""
        print(f"#{rank:<3} score={r['score']:.4f}{marker}")
        print(f"    {r['text'][:300]}{'...' if len(r['text']) > 300 else ''}")
        print()


if __name__ == "__main__":
    main()
