"""
Regenerate the cost-per-question trend table from traces/costs.jsonl
(appended to by every single call_llm() call in src/llm.py) without
re-running the pipeline. This is the "report your cost per question in
tokens and rupees, and show the trend across your eight questions"
deliverable.

    python scripts/cost_report.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import COST_LOG_PATH, USD_TO_INR


def main():
    path = Path(COST_LOG_PATH)
    if not path.exists():
        print(f"No cost log at {path} yet -- run scripts/run_pipeline.py first.")
        return

    by_question = defaultdict(lambda: {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "search_calls": 0, "by_stage": defaultdict(float)})
    order = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        qid = rec["question_id"]
        if qid not in by_question:
            order.append(qid)
        b = by_question[qid]
        b["cost_usd"] += rec["cost_usd"]
        b["input_tokens"] += rec["input_tokens"]
        b["output_tokens"] += rec["output_tokens"]
        b["search_calls"] += rec["search_calls"]
        b["by_stage"][rec["stage"]] += rec["cost_usd"]

    print(f"{'question':<10} {'cost (USD)':>12} {'cost (INR)':>12} {'in_tok':>8} {'out_tok':>8} {'search':>7}   by stage")
    print("-" * 100)
    prev_cost = None
    for qid in order:
        b = by_question[qid]
        trend = ""
        if prev_cost is not None and prev_cost > 0:
            pct = (b["cost_usd"] - prev_cost) / prev_cost * 100
            trend = f"  ({pct:+.0f}% vs prev)"
        stage_str = ", ".join(f"{s}=${c:.4f}" for s, c in sorted(b["by_stage"].items(), key=lambda x: -x[1]))
        print(f"{qid:<10} {b['cost_usd']:>12.4f} {b['cost_usd']*USD_TO_INR:>12.2f} "
              f"{b['input_tokens']:>8} {b['output_tokens']:>8} {b['search_calls']:>7}   {stage_str}{trend}")
        prev_cost = b["cost_usd"]

    total = sum(b["cost_usd"] for b in by_question.values())
    print("-" * 100)
    print(f"{'TOTAL':<10} {total:>12.4f} {total*USD_TO_INR:>12.2f}")
    print(f"\n(USD_TO_INR = {USD_TO_INR} -- set/verify against the day's actual rate in .env before citing this in DECISIONS.md)")


if __name__ == "__main__":
    main()
