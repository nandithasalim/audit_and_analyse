"""
For every "unsupported" claim on a question, trace a specific date/fact
mentioned in the claim back through the pipeline to find out where it
actually came from (or didn't):

  1. The RAW retrieve() search summary for the query that produced this
     citation -- was the date in what the model found, before any
     filtering?
  2. The full fetched page the auditor pulled for this citation (from
     data/memory.db, where the auditor caches fetched pages) -- is the
     date anywhere on the real page at all, even outside the top-ranked
     chunks?

This tells us, for real, whether synthesize() invented a plausible-looking
date that was never in the evidence (a precision-hallucination risk), or
whether the date was genuinely there and something downstream (chunk
selection) just didn't surface it.

    python scripts/trace_claim_origin.py q05
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import memory
from src.config import TRACES_DIR

_DATE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:[\u2013\-]\d{1,2})?,?\s+\d{4}"
)


def _find_date(claim_text: str) -> str | None:
    m = _DATE_RE.search(claim_text)
    return m.group(0) if m else None


def main():
    if len(sys.argv) != 2:
        print("usage: python scripts/trace_claim_origin.py <question_id>")
        sys.exit(1)
    qid = sys.argv[1]

    traces_dir = Path(TRACES_DIR)
    audit_path = traces_dir / f"{qid}_audit.json"
    analyst_path = traces_dir / f"{qid}_analyst.json"
    if not audit_path.exists() or not analyst_path.exists():
        print(f"Missing trace files for {qid} in {traces_dir}")
        sys.exit(1)

    audit_data = json.loads(audit_path.read_text())
    analyst_data = json.loads(analyst_path.read_text())
    retrieval = analyst_data.get("retrieval", [])

    memory.init_db()

    flagged = [a for a in audit_data["claim_audits"] if a["verdict"] == "unsupported"]
    print(f"{qid}: tracing {len(flagged)} unsupported claim(s)\n" + "=" * 80)

    for i, a in enumerate(flagged, start=1):
        claim = a["claim_text"]
        url = a["citation_url"]
        date = _find_date(claim)
        print(f"\n--- [{i}] CLAIM: {claim}")
        print(f"    citation: {url}")
        if not date:
            print("    (no clear date pattern found in this claim -- skipping date trace)")
            continue
        print(f"    date to trace: {date!r}")

        # 1. was this date in the RAW retrieve() summary that cited this url?
        matching_retrievals = [r for r in retrieval if any(c.get("url") == url for c in r.get("citations", []))]
        if not matching_retrievals:
            print(f"    [retrieve()] no retrieval item found whose citations include this url"
                  f" (citation may have been attached to a different query's result)")
        for r in matching_retrievals:
            found = date in r["text"]
            print(f"    [retrieve() raw summary, query={r['query']!r}] date present: {found}")
            if not found:
                print(f"        raw summary was: {r['text'][:600]}")

        # 2. is the date anywhere on the FULL fetched page (not just the top-5 chunks)?
        doc = memory.get_document(url)
        if doc is None:
            print(f"    [auditor's fetched page] not found in data/memory.db -- fetch may not have been cached, or db was reset")
        else:
            found_on_page = date in doc["raw_text"]
            print(f"    [full fetched page] date present anywhere on page: {found_on_page}")
            if found_on_page:
                idx = doc["raw_text"].index(date)
                print(f"        context: ...{doc['raw_text'][max(0, idx-100):idx+150]}...")


if __name__ == "__main__":
    main()
