"""
Run the analyst, then the auditor, then (if the auditor flagged anything)
the repair gate, across the 8 questions in questions/questions.yaml -- in
order, so memory built up by an earlier question is available to a later
one that reuses its entities.

    python scripts/run_pipeline.py                  # all 8 questions
    python scripts/run_pipeline.py --limit 2         # just the first 2 (cheap smoke test)
    python scripts/run_pipeline.py --only q05        # a single question by id
    python scripts/run_pipeline.py --skip-repair      # analyst + audit only
    python scripts/run_pipeline.py --fresh-memory      # wipe data/memory.db first (re-run all 8 as if for the first time)

Writes one trace file per question per stage to traces/<question_id>_<stage>.json,
and prints (and appends to traces/report.jsonl) a per-question summary line.
Cost-per-question numbers come from traces/costs.jsonl, which src/llm.py
appends to on every single model call -- run scripts/cost_report.py any
time to regenerate the cost trend table from that log without re-running
anything.
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src import memory
from src.analyst import run_question
from src.auditor import audit_answer
from src.config import DB_PATH, TRACES_DIR
from src.repair import repair_answer


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="questions/questions.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="run a single question id, e.g. q05")
    ap.add_argument("--skip-repair", action="store_true")
    ap.add_argument("--fresh-memory", action="store_true", help="wipe data/memory.db before running")
    args = ap.parse_args()

    if args.fresh_memory:
        db_path = Path(DB_PATH)
        if db_path.exists():
            db_path.unlink()
        print(f"Wiped {DB_PATH}")

    memory.init_db()

    questions = yaml.safe_load(open(args.questions))
    if args.only:
        questions = [q for q in questions if q["id"] == args.only]
        if not questions:
            print(f"No question with id={args.only}")
            sys.exit(1)
    if args.limit:
        questions = questions[: args.limit]

    traces_dir = Path(TRACES_DIR)
    report_path = traces_dir / "report.jsonl"
    traces_dir.mkdir(parents=True, exist_ok=True)

    for q in questions:
        qid, qtext = q["id"], q["question"].strip()
        print(f"\n=== {qid} (difficulty {q.get('difficulty', '?')}) ===")
        print(qtext)

        t0 = time.time()
        analyst_result = run_question(qtext, qid)
        _write_json(traces_dir / f"{qid}_analyst.json", dataclasses.asdict(analyst_result))
        print(f"\n[analyst] {analyst_result.answer}")
        print(f"[analyst] claims={len(analyst_result.claims)} known_facts_used={len(analyst_result.known_facts_used)} "
              f"facts_saved={analyst_result.facts_saved} cost=${analyst_result.cost_usd:.4f} "
              f"wall_clock={analyst_result.wall_clock_s:.1f}s")

        audit_result = audit_answer(analyst_result.answer, analyst_result.claims, qid)
        _write_json(traces_dir / f"{qid}_audit.json", dataclasses.asdict(audit_result))
        summary = audit_result.summary()
        print(f"[audit] {summary} cost=${audit_result.cost_usd:.4f} wall_clock={audit_result.wall_clock_s:.1f}s")

        repair_result = None
        needs_repair = summary["unsupported"] + summary["contradicted"] > 0
        if needs_repair and not args.skip_repair:
            repair_result = repair_answer(analyst_result, audit_result)
            _write_json(traces_dir / f"{qid}_repair.json", dataclasses.asdict(repair_result))
            print(f"[repair] attempted {len(repair_result.repaired_claims)} claim(s), cost=${repair_result.cost_usd:.4f}")
            print(f"[repair] revised answer: {repair_result.repaired_answer}")

        total_cost = analyst_result.cost_usd + audit_result.cost_usd + (repair_result.cost_usd if repair_result else 0.0)
        record = {
            "question_id": qid, "difficulty": q.get("difficulty"),
            "analyst_cost_usd": round(analyst_result.cost_usd, 6),
            "audit_cost_usd": round(audit_result.cost_usd, 6),
            "repair_cost_usd": round(repair_result.cost_usd, 6) if repair_result else 0.0,
            "total_cost_usd": round(total_cost, 6),
            "wall_clock_s": round(time.time() - t0, 1),
            "audit_summary": summary,
            "facts_saved": analyst_result.facts_saved,
            "known_facts_used": len(analyst_result.known_facts_used),
        }
        with open(report_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[total] ${total_cost:.4f}")

    print(f"\nDone. Per-question report appended to {report_path}")
    print("Run scripts/cost_report.py for the cost-per-question trend table.")


if __name__ == "__main__":
    main()
