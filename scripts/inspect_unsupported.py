"""
Look at the actual evidence behind every "unsupported" verdict for one
question, side by side: the claim, the auditor's reasoning/quote, and the
top-ranked evidence chunk it was actually judging against. Built to answer
one question honestly -- when the auditor says "unsupported", is that
because the source really doesn't say this, or because something upstream
(chunk selection, the naive scraper, the cheap judge model) missed it?

    python scripts/inspect_unsupported.py q05
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TRACES_DIR


def main():
    if len(sys.argv) != 2:
        print("usage: python scripts/inspect_unsupported.py <question_id>")
        sys.exit(1)
    qid = sys.argv[1]

    audit_path = Path(TRACES_DIR) / f"{qid}_audit.json"
    if not audit_path.exists():
        print(f"No audit trace at {audit_path}")
        sys.exit(1)

    data = json.loads(audit_path.read_text())
    flagged = [a for a in data["claim_audits"] if a["verdict"] == "unsupported"]

    if not flagged:
        print(f"{qid}: no 'unsupported' claims to inspect.")
        return

    print(f"{qid}: {len(flagged)} claim(s) marked 'unsupported'\n" + "=" * 80)
    for i, a in enumerate(flagged, start=1):
        print(f"\n--- [{i}] CLAIM: {a['claim_text']}")
        print(f"    citation: {a['citation_url']}")
        print(f"    auditor's quote: {a['quote']!r}")
        print(f"    auditor's reasoning: {a['reasoning']}")
        chunks = a.get("evidence_chunks") or []
        print(f"    ({len(chunks)} evidence chunks were considered; top-ranked one shown below)")
        if chunks:
            top = chunks[0]
            print(f"    top chunk (score={top['score']:.3f}): {top['text'][:500]}")
        else:
            print("    (no evidence chunks -- shouldn't happen for 'unsupported', flag this if you see it)")


if __name__ == "__main__":
    main()
