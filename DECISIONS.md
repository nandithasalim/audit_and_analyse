# DECISIONS.md

## Architecture

**Analyst**: `plan -> parallel retrieve -> resolve_evidence -> synthesize` (`src/analyst.py`), plain functions, not a graph framework. `plan()` checks memory before deciding what to search, so a later question about an already-researched entity gets a narrower plan instead of re-researching from zero. `retrieve()` fires one web-search-enabled call per planned sub-query in parallel (`ThreadPoolExecutor`). `resolve_evidence()` ranks each result's text against the question with embeddings (not TF-IDF -- see below) and tags each source's trust tier before handing evidence to `synthesize()`, the only step that produces citable claims -- it never calls web_search itself, so every claim traces back to one `retrieve()` call.

**Memory**: a typed fact graph (subject/relation/object/source_url) in SQLite, not an answer cache -- the point is transfer to a *different* question about the same entity. Retrieval is hybrid FTS5 (bm25) + TF-IDF for fuzzy entity-name overlap; unlike chunk selection, this stayed on TF-IDF since entity-name lookup doesn't suffer the same "stated differently than the claim" problem.

**Auditor**: independent by construction -- it never sees the analyst's reasoning, only the claim, and fetches/ranks its own evidence (`src/fetch.py`, `src/retriever.py`). Five verdicts, not three (`supported | unsupported | contradicted | no_citation | unverifiable`), because collapsing "the source doesn't say this" and "the source couldn't be fetched" hides which failure actually happened.

**Repair**: one re-search, one rewrite, never re-audited -- an unbounded loop makes cost unpredictable and lets a claim converge on merely *sounding* well-cited.

## Cost/quality levers

- **Model tiering by task shape, not pipeline stage.** `plan()` stays on the strongest model (astra) -- a bad plan poisons everything downstream. `retrieve()` and repair's re-search run on a mid tier (terra, ~5x cheaper) -- "search and report back" doesn't need flagship reasoning. `synthesize()` runs one tier down (sol, half of astra) -- it resolves source disagreements via trust tiers, so not pure lookup, but not open-ended either. `audit_claim`, entity extraction, and the repair rewrite run on the cheapest tier (luna) -- bounded classification over already-narrowed input, and the highest call volume in the pipeline.
- **Semantic (embedding) chunk selection replaces TF-IDF**, for both the analyst's evidence filtering and the auditor's chunk ranking. The TF-IDF failure mode was reproduced offline, not just asserted: for "Tanishq is owned by Titan Company" against four synthetic chunks, the chunk that actually states ownership ranked #3 of 4, outranked by a keyword-dense chunk that never states the relationship (`tests/test_offline.py::test_retriever_known_limitation`, kept as documented before/after).
- **Source trust tiering** (`src/trust.py`) -- a heuristic domain allowlist (regulatory/official > established news > trade press > unranked), fed to `synthesize()` as one input when sources disagree, not an automatic override. A domain off the list is honestly "unranked," not falsely trusted.

## Real bugs found running the live batch

- `PRICING_PER_1M_TOKENS` was placeholder-guessed and never verified -- corrected before the paid run; every cost number before that understated real spend ~5-8x (OpenAI's bill was always correct; only what this code reported was wrong).
- `embed()` sent unbatched chunks in one call -- blew OpenAI's 300k-token-per-request cap on q08 (its cited page chunked to 1.6M+ tokens). Fixed by batching on `tiktoken`'s real token counts, not a character-count guess (a first guess was itself off ~2x on this dense content and still failed).
- `chunk_text()` claimed to hard-split oversized text but didn't, for a paragraph with no sentence punctuation (a data-table block) -- blew OpenAI's separate 8192-token per-item cap. Fixed the splitter to actually hard-split on length; added a defensive per-item clip in `embeddings.py`.
- `AnalystResult`/`AuditResult.cost_usd` never included embedding cost (logged correctly to `traces/costs.jsonl`, never added to the in-memory total) -- live per-question totals print low (q08: printed $1.07, actual $1.36). Billing was never affected, only the terminal print. Not fixed; noted since no reported number here relies on the live print.

## Real cost per question (`scripts/cost_report.py`)

| q | difficulty | cost (USD) | cost (INR) | pipeline |
|---|---|---|---|---|
| q01 | 1 | $0.6559 | Rs.57.72 | original (astra + TF-IDF) |
| q02 | 1 | $0.4165 | Rs.36.65 | original |
| q03 | 2 | $1.1935 | Rs.105.03 | original |
| q04 | 2 | $0.3685 | Rs.32.43 | this session's pipeline |
| q05 | 3 | $0.7402 | Rs.65.14 | this session's pipeline |
| q06 | 3 | $0.3724 | Rs.32.77 | this session's pipeline |
| q07 | 4 | $0.1954 | Rs.17.19 | this session's pipeline |
| q08 | 5 | $1.3597 | Rs.119.65 | this session's pipeline |

Total: **$5.30 / Rs.466.58** (excludes a $0.06 smoke test outside the question set; INR at 88/USD).

q01-q03 ran under the original pipeline before this session's tiering/embeddings/trust changes; q04-q08 ran after. q04 ran *last*, out of sequence (its original run was cut short by an OpenAI billing interruption, completed only once q05-08 were already done), so its `plan()` saw memory facts from later questions too -- disclosed rather than hidden. Raw cost does **not** decrease monotonically q1->q8; it tracks question difficulty and cited-page size (q04 and q08 both cite large Titan pages with outsized embedding-token counts, 7.7M and 14.7M) far more than question order. The cleaner evidence for memory transfer is `known_facts_used`, which *does* climb as entities recur: q05 (new domain) = 0, q06 = 3, q07 = 3, q08 (heaviest Titan/Tanishq reuse from q01 and q04) = 9 -- each one is memory answering part of the question for free.

## What the audit actually found

`supported` verdicts were rare across q05-q08 -- worth explaining rather than hiding. Two real, distinct causes, traced with `scripts/trace_claim_origin.py` and `scripts/inspect_unsupported.py` rather than guessed at:

1. **The auditor checks only the one cited page's body text, and specific facts (dates especially) often don't live there.** OpenAI's web_search tool aggregated a per-company date table across many sources for q05; `synthesize()` correctly attributed each date to that company's own article. But the auditor re-fetches only that one article's body -- and articles routinely omit the exact date from body prose (it's in a byline the scraper doesn't reach, or wasn't in that specific article). Traced concretely for 4 of 5 q05 "unsupported" claims: the date was real and present in the original search result, just not restatable from that one page's body alone. Not a hallucination -- a stricter, arguably *correct* standard for a citation-integrity check, but worth knowing about.
2. **Meta-reasoning gets audited as if it were a fact.** One q05 "claim" was the analyst explaining *why* it preferred one figure over another -- no external page could confirm an explanation of the analyst's own reasoning. A claim-extraction category error, not a factual one.

## Where it still breaks

The auditor's fetcher (`httpx` + `BeautifulSoup`) cannot read JS-rendered pages -- confirmed on a real run (a cited Titan page returned HTTP 200, zero extractable text), correctly reported as `unverifiable` rather than the more confident-sounding `unsupported`. Cross-source corroboration is per-`retrieve()`-result, not per-claim -- coarser than clustering individual facts, cut for time. Repair is never re-audited by design, so a wrong repair can ship uncaught.

## What's next with two more weeks

Playwright fallback for JS-rendered pages (render only when the cheap fetch returns ~0 chars). Exclude meta/self-referential claims from claim extraction. Let the auditor fall back to the original search result, not just the linked article, for facts that legitimately live outside one page's body. Real per-claim cross-source clustering instead of the per-`retrieve()`-call proxy. Fix the `cost_usd` accumulator gap so live totals match `cost_report.py` without needing the authoritative log.
