"""
Resume just the audit (+repair) stage for one question whose analyst run
already succeeded and was saved to traces/<qid>_analyst.json, but whose
audit step then crashed (e.g. the embeddings-batching bug fixed in
src/embeddings.py -- q08 hit exactly this: its cited page was too large to
embed in one request). Avoids re-paying for plan/retrieve/synthesize when
only the audit step needs re-running -- that cost already happened and is
already logged in traces/costs.jsonl from the original run.

    python scripts/resume_audit.py q08
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyst import AnalystResult
from src.auditor import audit_answer
from src.config import TRACES_DIR
from src.repair import repair_answer


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    if len(sys.argv) != 2:
        print("usage: python scripts/resume_audit.py <question_id>")
        sys.exit(1)
    qid = sys.argv[1]

    traces_dir = Path(TRACES_DIR)
    analyst_path = traces_dir / f"{qid}_analyst.json"
    if not analyst_path.exists():
        print(f"No saved analyst trace at {analyst_path} -- the analyst step for {qid} "
              f"never completed, so there's nothing to resume from. Run the full "
              f"pipeline for {qid} instead (python scripts/run_pipeline.py --only {qid}).")
        sys.exit(1)

    saved = json.loads(analyst_path.read_text())
    field_names = {f.name for f in dataclasses.fields(AnalystResult)}
    analyst_result = AnalystResult(**{k: v for k, v in saved.items() if k in field_names})
    print(f"Resuming {qid} from the saved analyst trace -- its analyst cost "
          f"(${analyst_result.cost_usd:.4f}) already happened in the original run "
          f"and is already in traces/costs.jsonl; not re-charging it.")
    print(f"\n[analyst, from saved trace] {analyst_result.answer}")

    t0 = time.time()
    audit_result = audit_answer(analyst_result.answer, analyst_result.claims, qid)
    _write_json(traces_dir / f"{qid}_audit.json", dataclasses.asdict(audit_result))
    summary = audit_result.summary()
    print(f"[audit] {summary} cost=${audit_result.cost_usd:.4f} wall_clock={audit_result.wall_clock_s:.1f}s")

    repair_result = None
    needs_repair = summary["unsupported"] + summary["contradicted"] > 0
    if needs_repair:
        repair_result = repair_answer(analyst_result, audit_result)
        _write_json(traces_dir / f"{qid}_repair.json", dataclasses.asdict(repair_result))
        print(f"[repair] attempted {len(repair_result.repaired_claims)} claim(s), cost=${repair_result.cost_usd:.4f}")
        print(f"[repair] revised answer: {repair_result.repaired_answer}")

    resumed_cost = audit_result.cost_usd + (repair_result.cost_usd if repair_result else 0.0)
    record = {
        "question_id": qid, "difficulty": None,
        "analyst_cost_usd": round(analyst_result.cost_usd, 6),  # from the original run, not re-charged here
        "audit_cost_usd": round(audit_result.cost_usd, 6),
        "repair_cost_usd": round(repair_result.cost_usd, 6) if repair_result else 0.0,
        "total_cost_usd": round(analyst_result.cost_usd + resumed_cost, 6),
        "wall_clock_s": round(time.time() - t0, 1),
        "audit_summary": summary,
        "facts_saved": analyst_result.facts_saved,
        "known_facts_used": len(analyst_result.known_facts_used),
        "resumed_from_saved_analyst": True,
    }
    report_path = traces_dir / "report.jsonl"
    with open(report_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\n[total for {qid}] ${record['total_cost_usd']:.4f} "
          f"(${analyst_result.cost_usd:.4f} analyst from the original run + ${resumed_cost:.4f} audit/repair just now)")
    print("scripts/cost_report.py will already show this correctly -- it reads traces/costs.jsonl directly, "
          "which call_llm() appended to during both the original run and this resume, so nothing is double-counted.")


if __name__ == "__main__":
    main()
