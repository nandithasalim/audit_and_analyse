"""
Export a Claude Code / Cowork session transcript (the raw .jsonl the CLI
keeps under ~/.claude/projects/) into a readable markdown log for the
/logs directory the brief requires ("export the full transcript of every
AI coding session and include it in your repository under /logs").

    python scripts/export_logs.py                       # auto-detect this project's transcripts
    python scripts/export_logs.py --input <path.jsonl>   # export one specific transcript
    python scripts/export_logs.py --list                 # just list candidate transcripts, export nothing

Auto-detect looks under ~/.claude/projects/ for a directory matching this
project's path and exports every .jsonl session found there. If you also
used a *different* tool (Cursor, Copilot, a different machine) for part of
this project, export those separately with that tool's own means -- this
script only knows about Claude Code / Cowork transcripts.

IMPORTANT: skim the exported file before submitting. This strips system-
reminder boilerplate and truncates long tool results for readability, but
it does not redact anything -- if a transcript has an API key or other
secret pasted into it anywhere, remove that line by hand before committing
/logs.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _find_transcripts() -> list[Path]:
    project_dir = Path(__file__).resolve().parent.parent
    # Claude Code encodes the project path into the transcripts directory name
    # by replacing "/" with "-".
    encoded = str(project_dir).replace("/", "-")
    candidates = []
    base = Path.home() / ".claude" / "projects"
    if base.exists():
        for d in base.iterdir():
            if d.is_dir() and (encoded in d.name or d.name in encoded):
                candidates.extend(sorted(d.glob("*.jsonl")))
        # also check the parent directory itself in case the encoding differs
        for f in base.rglob("*.jsonl"):
            if f not in candidates:
                candidates.append(f)
    return candidates


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "\n".join(parts)
    return ""


def _tool_uses(content) -> list[dict]:
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]


def _tool_results(content) -> list[dict]:
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]


def export(input_path: Path, output_path: Path, max_result_chars: int = 800) -> int:
    lines = [f"# Session transcript: `{input_path.name}`\n"]
    n_turns = 0

    with open(input_path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            t = obj.get("type")
            if t == "user":
                content = obj.get("message", {}).get("content")
                results = _tool_results(content)
                if results:
                    for r in results:
                        text = r.get("content")
                        if isinstance(text, list):
                            text = " ".join(x.get("text", "") for x in text if isinstance(x, dict))
                        text = str(text or "")
                        if len(text) > max_result_chars:
                            text = text[:max_result_chars] + f"... [truncated, {len(text)} chars total]"
                        lines.append(f"**Tool result:**\n```\n{text}\n```\n")
                    continue
                text = _text_of(content).strip()
                if text and not text.startswith("<system-reminder>"):
                    n_turns += 1
                    lines.append(f"## User\n{text}\n")

            elif t == "assistant":
                content = obj.get("message", {}).get("content")
                text = _text_of(content).strip()
                if text:
                    lines.append(f"## Assistant\n{text}\n")
                for tu in _tool_uses(content):
                    name = tu.get("name", "?")
                    inp = json.dumps(tu.get("input", {}), default=str)
                    if len(inp) > 300:
                        inp = inp[:300] + "..."
                    lines.append(f"**Tool call:** `{name}`  \n`{inp}`\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return n_turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="path to one specific .jsonl transcript")
    ap.add_argument("--outdir", default="logs")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.input:
        targets = [Path(args.input)]
    else:
        targets = _find_transcripts()

    if not targets:
        print("No transcripts found. Pass --input <path.jsonl> explicitly, or check "
              "~/.claude/projects/ for the right session file.")
        return

    if args.list:
        for t in targets:
            print(t)
        return

    outdir = Path(args.outdir)
    for t in targets:
        out = outdir / f"{t.stem}.md"
        n = export(t, out)
        print(f"Exported {t} -> {out} ({n} user turns)")

    print("\nReview the exported file(s) before committing -- see this script's docstring re: secrets.")


if __name__ == "__main__":
    main()
