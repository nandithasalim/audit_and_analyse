# Dyla Analyst & Auditor

Research agent (analyst) + independent claim-verification agent (auditor),
built for Dyla's Problem 3 take-home.

## Setup (should take under 5 minutes, on a machine with normal internet)

```bash
python -m venv .venv && source .venv/bin/activate      # or your preferred env tool
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your real OPENAI_API_KEY
python scripts/smoke_test.py
```

If the smoke test prints an answer with real citation URLs and a non-zero
cost line, your setup is good and the rest of the pipeline can build on it.

**This must run somewhere with unrestricted outbound HTTPS.** It was built
inside a sandboxed cloud container whose org-level egress policy blocks
essentially everything, including `api.openai.com` itself (see
DECISIONS.md) — so it was authored and offline-tested there, but the
live run below has to happen on a normal machine or CI runner.

## Running the pipeline

```bash
# all 8 questions: analyst -> auditor -> repair (only for flagged claims)
python scripts/run_pipeline.py

# quick/cheap smoke test: just the first question
python scripts/run_pipeline.py --limit 1

# a single question by id, e.g. to re-check q07 (the disagreement case)
python scripts/run_pipeline.py --only q07

# re-run everything as if memory were empty (drops data/memory.db first)
python scripts/run_pipeline.py --fresh-memory
```

Each question writes its analyst/audit/repair trace to `traces/<id>_*.json`
and a one-line summary to `traces/report.jsonl`. Cost is logged per model
call (tagged by pipeline stage) to `traces/costs.jsonl` as it runs.

```bash
# cost-per-question trend table (USD + INR), regenerated from traces/costs.jsonl
python scripts/cost_report.py

# readable per-question Markdown report (Mermaid flow + evidence table) plus
# an overview.md trend across all questions -- written to traces/reports/
python scripts/render_report.py --all

# see the full TF-IDF ranking for a claim against a fetched page, not just
# the top-5 the auditor actually gets -- useful for the retrieval
# limitation documented in DECISIONS.md
python scripts/inspect_retrieval.py --url "https://en.wikipedia.org/wiki/Tanishq" \
    --claim "Tanishq is owned by Titan Company"
```

### Diagnostics (used to investigate real audit results, not part of the required run)

```bash
# if only the audit/repair stage of a question crashed, re-run just that
# stage from the saved analyst trace instead of re-paying for analyst too
python scripts/resume_audit.py q08

# for one question, print every claim the auditor marked "unsupported"
# alongside its quote/reasoning and top evidence chunk
python scripts/inspect_unsupported.py q05

# trace whether a specific date/fact in a flagged claim was actually present
# in the raw retrieve() search summary and the auditor's fetched page (no
# LLM calls -- pure regex/lookup against the saved traces and data/memory.db)
python scripts/trace_claim_origin.py q05
```

## Offline tests (no API key, no internet needed)

```bash
OPENAI_API_KEY=dummy python tests/test_offline.py
```

Covers JSON-extraction robustness, chunking, the memory store's hybrid
FTS5+TF-IDF retrieval end-to-end, and reproduces the TF-IDF chunk-selection
limitation with realistic synthetic evidence. Anything that actually calls
an LLM (planning, retrieval, synthesis, auditing, repair) needs a real key
and real internet — that's `scripts/run_pipeline.py`, not this.

## Exporting session logs (for `/logs`)

```bash
python scripts/export_logs.py
```

Auto-detects this project's Claude Code / Cowork session transcript(s)
under `~/.claude/projects/` and converts each to a readable markdown file
under `logs/`. Review the output before committing — see the script's
docstring.

## Project layout 
src/
config.py model routing (STRONG_MODEL / CHEAP_MODEL) + pricing table
llm.py the one call_llm() wrapper every stage uses (cost logging by stage)
jsonutil.py robust "extract JSON from an LLM text response" helper
memory.py typed fact graph (SQLite) + hybrid FTS5/TF-IDF retrieval
fetch.py independent page fetch + chunking, used only by the auditor
embeddings.py OpenAI embeddings wrapper (real token-budgeted batching)
retriever.py semantic (embedding) evidence-chunk ranking; the old TF-IDF
ranker is kept alongside it only for the offline test that
reproduces its documented known limitation
trust.py heuristic source-trust tiering, fed to synthesize()
facts.py claim -> (subject, relation, object) triple extraction
analyst.py plan -> parallel retrieve -> resolve_evidence -> synthesize
auditor.py independent 5-outcome claim verification
repair.py bounded single-pass repair for flagged claims
questions/questions.yaml the 8 questions (increasing difficulty, entity reuse)
scripts/ run_pipeline.py, cost_report.py, render_report.py, inspect_retrieval.py,
export_logs.py, smoke_test.py, plus diagnostics (resume_audit.py,
inspect_unsupported.py, trace_claim_origin.py -- see above)
tests/ offline tests (no network/key needed)
traces/ run traces + cost log + rendered reports (gitignored except .gitkeep)
logs/ exported AI coding session transcripts (see above)
data/ memory.db (gitignored)

## Status

- [x] Project scaffold + config
- [x] LLM wrapper with per-stage cost logging
- [x] Memory store (typed relationship graph + hybrid FTS/TF-IDF)
- [x] Analyst graph (plan -> parallel retrieve -> resolve_evidence -> synthesize)
- [x] Auditor (independent citation check: supported/unsupported/contradicted/no_citation/unverifiable)
- [x] Repair gate (RARR-style, bounded single pass)
- [x] 8 questions + run/eval scripts
- [x] A real end-to-end run on a machine with internet, all 8 questions ($5.30 / Rs.466.58 total)
- [x] DECISIONS.md (architecture, real cost table, real audit findings, known limitations)

See DECISIONS.md for architecture rationale, trade-offs, and known limitations.
