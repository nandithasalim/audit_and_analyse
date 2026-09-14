"""
RARR-style repair gate, bounded to a single pass.

Given an analyst answer and the auditor's verdicts on it, attempt to fix
exactly the claims marked "unsupported" or "contradicted": one fresh,
narrowly-targeted web search per flagged claim, then one rewrite of the
final answer paragraph to incorporate whatever the re-search found (or to
soften/remove the claim if the re-search still can't back it up).

Bounded deliberately: repaired claims are NOT re-audited. Re-auditing and
re-repairing until everything passes is exactly the unbounded loop that
would make cost unpredictable and would let the system quietly "fix" a
claim by citing a source that merely sounds authoritative rather than one
that's actually correct -- the auditor would eventually rubber-stamp
whatever the loop converges to. One attempt, then stop and report
"attempted, not reverified" honestly, is a smaller promise but one this
pipeline can actually keep. If you disagree, run the auditor again on the
repaired output yourself (scripts/run_pipeline.py --audit-only) and look
at the numbers -- that's the honest way to see whether repair actually
worked, rather than trusting the same loop to grade its own fix.
"""
import time
from dataclasses import dataclass, field
from typing import Optional

from src.analyst import AnalystResult
from src.auditor import AuditResult, ClaimAudit
from src.config import CHEAP_MODEL, STRONG_MODEL
from src.jsonutil import parse_json_response, JSONParseError
from src.llm import call_llm

_REPAIR_SEARCH_PROMPT = """An independent fact-checker flagged this claim as "{verdict}":

Claim: {claim}
Fact-checker's reasoning: {reasoning}

Search the web to find out whether this claim is actually correct. Return JSON:
{{
  "resolution": "confirmed" | "corrected" | "unresolvable",
  "text": "the corrected claim text if resolution is 'confirmed' or 'corrected' (may be identical to the original if confirmed), or the original claim text if 'unresolvable'",
  "citation_url": "a URL that actually supports the text above, or null if unresolvable"
}}

Rules:
- "confirmed": you found a source that does support the original claim as stated (the auditor may have just missed it).
- "corrected": the original claim was wrong or imprecise; you found the accurate version.
- "unresolvable": you could not find reliable support either way.
- Return ONLY the JSON object.
"""

_PATCH_ANSWER_PROMPT = """Rewrite this research answer to incorporate the following fixes, while keeping everything else about it the same. Keep it concise (2-6 sentences).

Original answer: {answer}

Fixes:
{fixes}

Return ONLY the revised answer text, no JSON, no preamble.
"""


@dataclass
class RepairedClaim:
    original_claim: str
    original_verdict: str
    resolution: str  # confirmed | corrected | unresolvable
    new_text: str
    new_citation_url: Optional[str]


@dataclass
class RepairResult:
    question_id: str
    original_answer: str
    repaired_answer: str
    repaired_claims: list[RepairedClaim] = field(default_factory=list)
    cost_usd: float = 0.0
    wall_clock_s: float = 0.0


def repair_answer(analyst_result: AnalystResult, audit_result: AuditResult) -> RepairResult:
    t0 = time.time()
    flagged = [a for a in audit_result.claim_audits if a.verdict in ("unsupported", "contradicted")]

    result = RepairResult(
        question_id=analyst_result.question_id,
        original_answer=analyst_result.answer,
        repaired_answer=analyst_result.answer,
    )

    if not flagged:
        return result

    for audit in flagged:
        r = call_llm(
            _REPAIR_SEARCH_PROMPT.format(verdict=audit.verdict, claim=audit.claim_text, reasoning=audit.reasoning),
            question_id=analyst_result.question_id, stage="repair", model=STRONG_MODEL, use_web_search=True,
        )
        result.cost_usd += r.cost_usd
        try:
            parsed = parse_json_response(r.text)
        except JSONParseError:
            parsed = {"resolution": "unresolvable", "text": audit.claim_text, "citation_url": None}

        result.repaired_claims.append(RepairedClaim(
            original_claim=audit.claim_text,
            original_verdict=audit.verdict,
            resolution=parsed.get("resolution", "unresolvable"),
            new_text=parsed.get("text", audit.claim_text),
            new_citation_url=parsed.get("citation_url"),
        ))

    fixes_str = "\n".join(
        f"- Was flagged {rc.original_verdict}: \"{rc.original_claim}\" -> "
        f"{'REMOVE this claim, could not confirm it' if rc.resolution == 'unresolvable' else rc.new_text}"
        for rc in result.repaired_claims
    )
    r = call_llm(
        _PATCH_ANSWER_PROMPT.format(answer=analyst_result.answer, fixes=fixes_str),
        question_id=analyst_result.question_id, stage="repair", model=CHEAP_MODEL,
    )
    result.cost_usd += r.cost_usd
    result.repaired_answer = r.text.strip()

    result.wall_clock_s = time.time() - t0
    return result
