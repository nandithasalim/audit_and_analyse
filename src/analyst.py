"""
The analyst: plan -> parallel retrieve -> resolve_evidence -> synthesize.

Design notes (see DECISIONS.md for the full rationale):

- plan() first asks memory what it already knows about the entities in the
  question, and tells the planning model about those known facts so it can
  write narrower search queries (or none at all, for a fact already on
  file) instead of re-researching an entity from zero every time. This is
  the mechanism behind "a later question about an entity it has already
  researched is answered faster and better" -- it is memory that transfers
  across questions, not an answer cache keyed on the literal question text.

- retrieve() fires one call_llm(..., use_web_search=True) per planned
  sub-query, in parallel via a thread pool. call_llm is a blocking HTTP
  call under the hood (OpenAI's python client), so threads -- not asyncio --
  give real wall-clock parallelism here for cheap.

- resolve_evidence() is a thin cross-check pass: for each sub-query result
  it counts distinct source domains in the citations OpenAI's web_search
  returned, and flags results backed by a single domain as single_source.
  This is a coarse proxy for "cross-check claims that appear in only one
  source" (true per-claim corroboration would mean clustering individual
  factual statements across results, which is a lot more machinery for a
  take-home) -- synthesize() is told to hedge language on single_source
  evidence and is allowed to note the gap in its answer.

- synthesize() is the only step that produces citable, user-facing claims.
  It does NOT call web_search itself -- it works only from the evidence
  handed to it, so "where did this claim's citation come from" always has
  one answer: a specific retrieve() call, traceable in the run log.
"""
import concurrent.futures
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from src.config import CHEAP_MODEL, SEARCH_MODEL, STRONG_MODEL, SYNTH_MODEL
from src.facts import extract_facts_from_claim
from src.fetch import chunk_text
from src.jsonutil import parse_json_response, JSONParseError
from src.llm import call_llm
from src import memory
from src.retriever import select_relevant_chunks
from src.trust import tag_citations

_ENTITY_PROMPT = """List the specific named entities (companies, people, products) mentioned or clearly implied in this research question. Return ONLY a JSON array of strings, e.g. ["Tanishq", "Titan Company"]. If none, return [].

Question: {question}
"""

_PLAN_PROMPT = """You are planning how to research this question using live web search.

Question: {question}

Facts already known from previous research (may be empty, may be partial, may be irrelevant -- use your judgement):
{known_facts}

Write a research plan as JSON:
{{
  "search_queries": ["specific query 1", "specific query 2", ...],
  "plan_notes": "one or two sentences on your approach, and what (if anything) you're skipping because it's already known"
}}

Rules:
- Each search_query should be a specific, self-contained web search query (not a restatement of the whole question).
- Default to 2-3 targeted queries. Only go to 4-5 when the question calls for a specific number, date, or ranking that a single source might get wrong or that outlets commonly disagree on -- each extra query is a real added cost, so add one only when cross-checking that specific fact is worth it, not by default "for safety."
- If known facts already fully answer the question, you may return an empty search_queries list and say so in plan_notes.
- Return ONLY the JSON object.
"""

_SYNTHESIZE_PROMPT = """Answer the research question using ONLY the evidence below. Do not use outside knowledge and do not guess.

Question: {question}

Evidence (from web search and prior memory):
{evidence}

Return JSON:
{{
  "answer": "the full prose answer, 2-6 sentences, written for a reader who has not seen the evidence",
  "claims": [
    {{"text": "one atomic factual statement from the answer", "citation_url": "https://... or null if you cannot point to a specific source for it"}}
  ]
}}

Rules:
- Every specific factual statement in "answer" (a number, a name, a date, a ranking) must also appear as an entry in "claims" with its supporting citation_url.
- Each evidence block below is tagged with a trust tier for its source(s): tier 1 = official/regulatory source, tier 2 = established news, tier 3 = trade/industry press, tier 4 = unranked/unrecognized. When sources disagree on a fact, prefer the higher-trust-tier source, but treat tier as ONE input, not an automatic override -- a very recent tier-4 source can still be right over a stale tier-2 one, and you should weigh recency and specificity too. State which source you preferred and briefly why (trust tier, recency, or specificity) rather than picking silently or reporting both numbers and shrugging.
- If the evidence does not actually answer the question (or part of it), say so explicitly in the answer ("I could not find ...") rather than filling the gap with a plausible-sounding guess. Do not invent a citation to cover a gap.
- Return ONLY the JSON object.
"""


@dataclass
class AnalystResult:
    question_id: str
    question: str
    answer: str = ""
    claims: list[dict] = field(default_factory=list)
    plan_notes: str = ""
    entities: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    known_facts_used: list[dict] = field(default_factory=list)
    retrieval: list[dict] = field(default_factory=list)
    facts_saved: int = 0
    cost_usd: float = 0.0
    wall_clock_s: float = 0.0
    trace: list[dict] = field(default_factory=list)


def _log(result: AnalystResult, step: str, detail: dict) -> None:
    result.trace.append({"step": step, "t": time.time(), **detail})


def _extract_entities(question: str, question_id: str) -> list[str]:
    r = call_llm(
        _ENTITY_PROMPT.format(question=question),
        question_id=question_id, stage="plan", model=CHEAP_MODEL,
    )
    try:
        parsed = parse_json_response(r.text)
        if isinstance(parsed, list):
            return [str(e) for e in parsed if str(e).strip()]
    except JSONParseError:
        pass
    return []


def _domains(citations: list[dict]) -> set[str]:
    out = set()
    for c in citations:
        url = c.get("url", "")
        try:
            out.add(urlparse(url).netloc)
        except Exception:
            continue
    return out


def plan(question: str, question_id: str, result: AnalystResult) -> dict:
    entities = _extract_entities(question, question_id)
    result.entities = entities
    known_facts: list[dict] = []
    for entity in entities:
        known_facts.extend(memory.get_facts_for_entity(entity, top_k=5))
    # dedupe by fact id
    seen = set()
    deduped = []
    for f in known_facts:
        if f["id"] in seen:
            continue
        seen.add(f["id"])
        deduped.append(f)
    result.known_facts_used = deduped

    known_facts_str = "\n".join(
        f"- {f['subject']} {f['relation']} {f['object']} (source: {f.get('source_url') or 'unknown'})"
        for f in deduped
    ) or "(none)"

    r = call_llm(
        _PLAN_PROMPT.format(question=question, known_facts=known_facts_str),
        question_id=question_id, stage="plan", model=STRONG_MODEL,
    )
    result.cost_usd += r.cost_usd
    try:
        parsed = parse_json_response(r.text)
    except JSONParseError:
        parsed = {"search_queries": [question], "plan_notes": "plan parse failed, falling back to raw question"}

    queries = parsed.get("search_queries") or []
    result.search_queries = queries
    result.plan_notes = parsed.get("plan_notes", "")
    _log(result, "plan", {"entities": entities, "known_facts": len(deduped), "search_queries": queries, "plan_notes": result.plan_notes})
    return parsed


def retrieve(search_queries: list[str], question_id: str, result: AnalystResult) -> list[dict]:
    if not search_queries:
        return []

    def _run_one(q: str) -> dict:
        r = call_llm(q, question_id=question_id, stage="retrieve", model=SEARCH_MODEL, use_web_search=True)
        return {"query": q, "text": r.text, "citations": r.citations, "cost_usd": r.cost_usd, "search_calls": r.search_calls}

    retrieved = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(search_queries))) as ex:
        futures = {ex.submit(_run_one, q): q for q in search_queries}
        for fut in concurrent.futures.as_completed(futures):
            item = fut.result()
            result.cost_usd += item["cost_usd"]
            retrieved.append(item)
            _log(result, "retrieve", {"query": item["query"], "n_citations": len(item["citations"]), "search_calls": item["search_calls"]})

    # keep output order stable-ish (matches input order) for readability in traces
    order = {q: i for i, q in enumerate(search_queries)}
    retrieved.sort(key=lambda x: order.get(x["query"], 0))
    result.retrieval = retrieved
    return retrieved


def resolve_evidence(retrieved: list[dict], known_facts: list[dict], question: str, result: AnalystResult) -> str:
    lines = []
    for f in known_facts:
        lines.append(f"[memory] {f['subject']} {f['relation']} {f['object']} (source: {f.get('source_url') or 'unknown'})")

    for item in retrieved:
        domains = _domains(item["citations"])
        single_source = len(domains) <= 1
        item["single_source"] = single_source
        item["domains"] = sorted(domains)
        tag = "SINGLE-SOURCE" if single_source else f"corroborated across {len(domains)} sources"
        # trust.tag_citations appends a [tier N: label] to each source instead
        # of a plain title/url join, so synthesize() has the trust signal it
        # needs to resolve disagreements (see _SYNTHESIZE_PROMPT's rules)
        # without a separate LLM call to look sources up.
        cite_str = tag_citations(item["citations"], entity_hint=" ".join(result.entities))

        # TF-IDF-filter the retrieved text against the overall question before
        # it goes into synthesize()'s prompt, instead of passing the raw
        # web_search summary through unfiltered -- same technique the auditor
        # already uses on cited pages (src/retriever.py), applied here to
        # keep synthesize focused on the on-topic part of each result. NOTE:
        # for a short retrieve() summary (a few sentences) this chunks down
        # to only 1-2 pieces and filters almost nothing -- the real payoff
        # shows up on questions whose retrieve() results come back long
        # (e.g. q05's open-ended enumeration). Check traces/costs.jsonl's
        # "synthesize" stage cost before/after to see the actual effect
        # rather than assuming this alone is what moves the number.
        chunks = chunk_text(item["text"], url=item["query"])
        if chunks:
            top_chunks = select_relevant_chunks(question, chunks, top_k=3, question_id=result.question_id)
            text_for_synthesis = "\n".join(c["text"] for c in top_chunks) if top_chunks else item["text"]
        else:
            text_for_synthesis = item["text"]

        lines.append(f"[web search: \"{item['query']}\" -- {tag}]\n{text_for_synthesis}\nCited: {cite_str}")
        _log(result, "resolve_evidence", {
            "query": item["query"], "single_source": single_source, "domains": item["domains"],
            "chunks_total": len(chunks), "chunks_kept": len(chunks) and min(3, len(chunks)),
        })

    return "\n\n".join(lines) if lines else "(no evidence gathered)"


def synthesize(question: str, evidence_str: str, question_id: str, result: AnalystResult) -> None:
    r = call_llm(
        _SYNTHESIZE_PROMPT.format(question=question, evidence=evidence_str),
        question_id=question_id, stage="synthesize", model=SYNTH_MODEL,
    )
    result.cost_usd += r.cost_usd
    try:
        parsed = parse_json_response(r.text)
    except JSONParseError:
        parsed = {"answer": r.text, "claims": []}

    result.answer = parsed.get("answer", "")
    result.claims = parsed.get("claims", [])
    _log(result, "synthesize", {"answer": result.answer, "n_claims": len(result.claims)})


def save_facts_to_memory(result: AnalystResult, question_id: str) -> None:
    saved = 0
    for claim in result.claims:
        triples = extract_facts_from_claim(claim.get("text", ""), question_id=question_id)
        for t in triples:
            t["source_url"] = claim.get("citation_url")
        saved += memory.save_facts(triples, question_id=question_id)
    result.facts_saved = saved
    _log(result, "save_facts_to_memory", {"facts_saved": saved})


def run_question(question: str, question_id: str) -> AnalystResult:
    memory.init_db()
    result = AnalystResult(question_id=question_id, question=question)
    t0 = time.time()

    plan_out = plan(question, question_id, result)
    retrieved = retrieve(plan_out.get("search_queries", []), question_id, result)
    evidence_str = resolve_evidence(retrieved, result.known_facts_used, question, result)
    synthesize(question, evidence_str, question_id, result)
    save_facts_to_memory(result, question_id)

    result.wall_clock_s = time.time() - t0
    return result
