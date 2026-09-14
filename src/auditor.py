"""
The auditor: independently verifies an analyst answer, claim by claim.

"Independent" means concretely: it does not trust the analyst's synthesis
of what a source says -- it fetches the cited URL itself (src/fetch.py),
re-ranks the page's own text against the claim (src/retriever.py), and asks
a fresh model call to judge only from that re-derived evidence
(audit_claim). The analyst's prose answer is never shown to audit_claim --
only the one claim under test and the chunks pulled from the cited page.
That separation is what makes "supported" mean something: it is not the
same model call agreeing with its earlier self.

Two independent-verification steps, not three, contra the checklist
phrasing in a hurried early README pass -- this build merges "confirm
citation resolves" and "confirm source supports claim" into one
audit_claim call once the evidence chunks are in hand, because splitting
them into separate LLM calls doubled cost for no accuracy gain in testing
(a fetch that 404s or times out never reaches audit_claim at all; see
_audit_one). Verdicts:
  - "supported"    -- the retrieved chunks state what the claim asserts
  - "contradicted"  -- the retrieved chunks state something that conflicts with the claim
  - "unsupported"   -- the retrieved chunks neither confirm nor conflict (most often: the
                        right chunk wasn't retrieved, or the source genuinely doesn't say this)
  - "no_citation"    -- the claim carries no citation_url to check at all
  - "unverifiable"   -- the citation_url exists but the page could not be fetched
                        (404, timeout, blocked, paywalled, JS-rendered with no server text)

An auditor that only ever says "supported" is worthless -- see
DECISIONS.md for the actual supported/unsupported/contradicted counts
across the 8 questions, including the Tanishq/Titan case where this
pipeline returns "unsupported" for a claim that is in fact true, and why.
"""
import time
from dataclasses import dataclass, field
from typing import Optional

from src.config import CHEAP_MODEL
from src.fetch import chunk_text, fetch_url
from src.jsonutil import parse_json_response, JSONParseError
from src.llm import call_llm
from src import memory
from src.retriever import select_relevant_chunks

_FIND_UNCAPTURED_PROMPT = """Here is a research answer, and the list of claims already extracted from it with their citations.

Answer: {answer}

Already-extracted claims:
{claims}

List any ADDITIONAL atomic factual statements (a number, a name, a date, a ranking) present in the answer text that are NOT already covered by the list above. Return ONLY a JSON array of strings (the additional claim text), or [] if none.
"""

_AUDIT_PROMPT = """You are auditing one factual claim from a research answer. Judge ONLY from the evidence chunks below -- they were pulled from the page the claim cites. Do not use outside knowledge.

Claim: {claim}

Evidence chunks from the cited source ({url}):
{chunks}

Return JSON:
{{
  "verdict": "supported" | "unsupported" | "contradicted",
  "quote": "the exact sentence or phrase from the evidence chunks that most supports your verdict, or null if none of the chunks say anything relevant to this claim",
  "reasoning": "one or two sentences explaining the verdict"
}}

Rules:
- "supported": the evidence chunks state what the claim asserts (paraphrase is fine, invented numbers/facts are not).
- "contradicted": the evidence chunks state something that conflicts with the claim (e.g. a different number, a denial).
- "unsupported": the evidence chunks neither confirm nor conflict -- e.g. they're about the right topic but don't state this specific fact. This is the correct verdict when you're not sure, not "supported".
- Return ONLY the JSON object.
"""


@dataclass
class ClaimAudit:
    claim_text: str
    citation_url: Optional[str]
    verdict: str  # supported | unsupported | contradicted | no_citation | unverifiable
    reasoning: str = ""
    quote: Optional[str] = None
    evidence_chunks: list[dict] = field(default_factory=list)


@dataclass
class AuditResult:
    question_id: str
    claim_audits: list[ClaimAudit] = field(default_factory=list)
    cost_usd: float = 0.0
    wall_clock_s: float = 0.0
    trace: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        counts = {"supported": 0, "unsupported": 0, "contradicted": 0, "no_citation": 0, "unverifiable": 0}
        for a in self.claim_audits:
            counts[a.verdict] = counts.get(a.verdict, 0) + 1
        return counts


def _log(result: AuditResult, step: str, detail: dict) -> None:
    result.trace.append({"step": step, "t": time.time(), **detail})


def find_uncaptured_claims(answer: str, claims: list[dict], question_id: str) -> list[str]:
    if not answer.strip():
        return []
    claims_str = "\n".join(f"- {c.get('text','')}" for c in claims) or "(none extracted)"
    r = call_llm(
        _FIND_UNCAPTURED_PROMPT.format(answer=answer, claims=claims_str),
        question_id=question_id, stage="audit", model=CHEAP_MODEL,
    )
    try:
        parsed = parse_json_response(r.text)
        if isinstance(parsed, list):
            return [str(c) for c in parsed if str(c).strip()]
    except JSONParseError:
        pass
    return []


def _get_or_fetch_chunks(url: str) -> list:
    doc = memory.get_document(url)
    if doc is not None:
        text = doc["raw_text"]
    else:
        text = fetch_url(url)
        memory.save_document(url, title="", raw_text=text)
    return chunk_text(text, url)


def _audit_one(claim_text: str, citation_url: Optional[str], question_id: str, result: AuditResult) -> ClaimAudit:
    if not citation_url:
        audit = ClaimAudit(claim_text=claim_text, citation_url=None, verdict="no_citation",
                            reasoning="Claim carries no citation to verify against.")
        _log(result, "audit_claim", {"claim": claim_text, "verdict": "no_citation"})
        return audit

    try:
        chunks = _get_or_fetch_chunks(citation_url)
    except Exception as e:
        audit = ClaimAudit(claim_text=claim_text, citation_url=citation_url, verdict="unverifiable",
                            reasoning=f"Could not fetch cited source: {e}")
        _log(result, "audit_claim", {"claim": claim_text, "url": citation_url, "verdict": "unverifiable", "error": str(e)})
        return audit

    evidence = select_relevant_chunks(claim_text, chunks, top_k=5)
    if not evidence:
        audit = ClaimAudit(claim_text=claim_text, citation_url=citation_url, verdict="unverifiable",
                            reasoning="Cited page returned HTTP 200 but no extractable body text (commonly a JavaScript-rendered page whose content isn't in the server-sent HTML) -- could not verify against it.",
                            evidence_chunks=[])
        _log(result, "audit_claim", {"claim": claim_text, "url": citation_url, "verdict": "unverifiable", "reason": "no_chunks"})
        return audit

    chunks_str = "\n\n".join(f"[{c['index']}] {c['text']}" for c in evidence)
    r = call_llm(
        _AUDIT_PROMPT.format(claim=claim_text, url=citation_url, chunks=chunks_str),
        question_id=question_id, stage="audit", model=CHEAP_MODEL,
    )
    result.cost_usd += r.cost_usd
    try:
        parsed = parse_json_response(r.text)
    except JSONParseError:
        parsed = {"verdict": "unsupported", "quote": None, "reasoning": "Audit model output was not parseable JSON."}

    verdict = parsed.get("verdict", "unsupported")
    if verdict not in ("supported", "unsupported", "contradicted"):
        verdict = "unsupported"

    audit = ClaimAudit(
        claim_text=claim_text, citation_url=citation_url, verdict=verdict,
        reasoning=parsed.get("reasoning", ""), quote=parsed.get("quote"), evidence_chunks=evidence,
    )
    _log(result, "audit_claim", {"claim": claim_text, "url": citation_url, "verdict": verdict, "top_chunk_score": evidence[0]["score"] if evidence else None})
    return audit


def audit_answer(answer: str, claims: list[dict], question_id: str) -> AuditResult:
    memory.init_db()
    result = AuditResult(question_id=question_id)
    t0 = time.time()

    uncaptured = find_uncaptured_claims(answer, claims, question_id)
    all_claims = list(claims) + [{"text": c, "citation_url": None} for c in uncaptured]
    if uncaptured:
        _log(result, "find_uncaptured_claims", {"n_found": len(uncaptured), "claims": uncaptured})

    for c in all_claims:
        audit = _audit_one(c.get("text", ""), c.get("citation_url"), question_id, result)
        result.claim_audits.append(audit)

    result.wall_clock_s = time.time() - t0
    return result
