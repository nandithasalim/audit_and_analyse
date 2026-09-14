"""
Renders a readable Markdown report per question from the raw JSON traces --
a Mermaid flowchart of what the pipeline actually did (real query counts,
real verdict counts, not a generic diagram), plus a proper evidence table
(verdict, citation, quote, reasoning per claim) instead of a wall of JSON.
GitHub renders ```mermaid fences natively, so these read correctly straight
in the repo.

    python scripts/render_report.py --all            # every question that has traces
    python scripts/render_report.py --question q01   # just one

Writes to traces/reports/<qid>.md, plus traces/reports/overview.md (the
cost/verdict trend across all questions + which entities got reused where,
when run with --all).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TRACES_DIR

VERDICT_ICON = {
    "supported": "✅",
    "unsupported": "⚠️",
    "contradicted": "❌",
    "no_citation": "⭕",
    "unverifiable": "❔",
}


def _load(traces_dir: Path, qid: str, stage: str):
    path = traces_dir / f"{qid}_{stage}.json"
    if not path.exists():
        return None
    return json.load(open(path))


def _mermaid_label(text: str, max_len: int = 60) -> str:
    text = text.replace('"', "'").replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def render_question(traces_dir: Path, qid: str, question_text: str = "") -> str:
    analyst = _load(traces_dir, qid, "analyst")
    audit = _load(traces_dir, qid, "audit")
    repair = _load(traces_dir, qid, "repair")

    if analyst is None:
        return f"# {qid}\n\n_No analyst trace found -- run `python scripts/run_pipeline.py --only {qid}` first._\n"

    question = question_text or analyst["question"]
    verdict_counts = {}
    if audit:
        for a in audit["claim_audits"]:
            verdict_counts[a["verdict"]] = verdict_counts.get(a["verdict"], 0) + 1
    verdict_str = " ".join(f"{VERDICT_ICON.get(v,'?')}{n} {v}" for v, n in verdict_counts.items()) or "(not audited)"

    total_cost = analyst["cost_usd"] + (audit["cost_usd"] if audit else 0) + (repair["cost_usd"] if repair else 0)

    lines = [f"# {qid}: {question}\n"]

    lines.append("```mermaid")
    lines.append("flowchart TD")
    lines.append(f'    Q["{_mermaid_label(question)}"]')
    known = len(analyst.get("known_facts_used", []))
    nq = len(analyst.get("search_queries", []))
    lines.append(f'    Q --> P["Plan: {nq} search queries planned<br/>{known} known fact(s) reused from memory"]')
    lines.append(f'    P --> R["Retrieve: {nq} parallel web searches"]')
    single = sum(1 for r in analyst.get("retrieval", []) if r.get("single_source"))
    lines.append(f'    R --> RE["Resolve evidence: {single} single-source, {nq-single} corroborated"]')
    lines.append(f'    RE --> S["Synthesize: {len(analyst.get("claims", []))} claims, ${analyst["cost_usd"]:.4f}"]')
    if audit:
        lines.append(f'    S --> A["Audit: {verdict_str}"]')
        if repair:
            n_repaired = len(repair["repaired_claims"])
            lines.append(f'    A -->|"{n_repaired} claim(s) flagged"| RP["Repair: {n_repaired} attempted, ${repair["cost_usd"]:.4f}"]')
            lines.append('    RP --> END["Final answer"]')
        else:
            lines.append('    A -->|"all clear"| END["Final answer"]')
    else:
        lines.append('    S --> END["Final answer (not yet audited)"]')
    lines.append("```\n")

    final_answer = repair["repaired_answer"] if repair else analyst["answer"]
    lines.append("## Answer\n")
    lines.append(final_answer + "\n")
    if repair and repair["repaired_answer"] != repair["original_answer"]:
        lines.append(f"<details><summary>Original answer before repair</summary>\n\n{repair['original_answer']}\n\n</details>\n")

    if audit:
        lines.append("## Evidence\n")
        lines.append("| Verdict | Claim | Citation | Quote | Reasoning |")
        lines.append("|---|---|---|---|---|")
        for a in audit["claim_audits"]:
            icon = VERDICT_ICON.get(a["verdict"], "?")
            cite = f"[link]({a['citation_url']})" if a.get("citation_url") else "—"
            quote = (a.get("quote") or "—").replace("\n", " ").replace("|", "\\|")
            reasoning = (a.get("reasoning") or "").replace("\n", " ").replace("|", "\\|")
            claim = a["claim_text"].replace("|", "\\|")
            lines.append(f"| {icon} {a['verdict']} | {claim} | {cite} | {quote[:150]} | {reasoning[:200]} |")
        lines.append("")

    if repair and repair["repaired_claims"]:
        lines.append("## Repair attempts\n")
        for rc in repair["repaired_claims"]:
            lines.append(f"- **{rc['resolution']}** (was `{rc['original_verdict']}`): "
                          f"\"{rc['original_claim']}\" -> \"{rc['new_text']}\""
                          + (f" ([source]({rc['new_citation_url']}))" if rc.get("new_citation_url") else ""))
        lines.append("")

    lines.append("## Cost\n")
    lines.append(f"- Analyst: ${analyst['cost_usd']:.4f}")
    if audit:
        lines.append(f"- Audit: ${audit['cost_usd']:.4f}")
    if repair:
        lines.append(f"- Repair: ${repair['cost_usd']:.4f}")
    lines.append(f"- **Total: ${total_cost:.4f}**")

    return "\n".join(lines) + "\n"


def render_overview(traces_dir: Path, qids: list[str]) -> str:
    report_path = traces_dir / "report.jsonl"
    records = []
    if report_path.exists():
        for line in open(report_path):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    by_qid = {r["question_id"]: r for r in records}

    lines = ["# Run overview\n", "## Cost and audit trend across questions\n"]
    lines.append("| Question | Difficulty | Known facts reused | Total cost | Audit verdicts |")
    lines.append("|---|---|---|---|---|")
    for qid in qids:
        r = by_qid.get(qid)
        if not r:
            continue
        vs = " ".join(f"{VERDICT_ICON.get(v,'?')}{n}" for v, n in r["audit_summary"].items() if n)
        lines.append(f"| {qid} | {r.get('difficulty','?')} | {r['known_facts_used']} | ${r['total_cost_usd']:.4f} | {vs} |")

    lines.append("\n## Memory reuse across questions\n")
    lines.append("```mermaid")
    lines.append("flowchart LR")
    for qid in qids:
        analyst = _load(traces_dir, qid, "analyst")
        if not analyst:
            continue
        label = f"{qid}<br/>{analyst['facts_saved']} facts saved"
        lines.append(f'    {qid}["{label}"]')
        if analyst.get("known_facts_used"):
            sources = {f.get("question_id") for f in analyst["known_facts_used"] if f.get("question_id")}
            for src in sources:
                if src != qid:
                    lines.append(f"    {src} -.->|reused| {qid}")
    lines.append("```\n")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--questions-file", default="questions/questions.yaml")
    args = ap.parse_args()

    traces_dir = Path(TRACES_DIR)
    out_dir = traces_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    qmeta = {q["id"]: q["question"].strip() for q in yaml.safe_load(open(args.questions_file))}

    if args.question:
        qids = [args.question]
    else:
        qids = sorted(p.stem.rsplit("_", 1)[0] for p in traces_dir.glob("*_analyst.json"))

    for qid in qids:
        md = render_question(traces_dir, qid, qmeta.get(qid, ""))
        out_path = out_dir / f"{qid}.md"
        out_path.write_text(md)
        print(f"wrote {out_path}")

    if args.all or len(qids) > 1:
        overview_md = render_overview(traces_dir, sorted(qmeta.keys()))
        (out_dir / "overview.md").write_text(overview_md)
        print(f"wrote {out_dir / 'overview.md'}")


if __name__ == "__main__":
    main()
