"""
extract_facts_from_claim: turns a synthesized claim sentence into zero or
more (subject, relation, object) triples for the memory graph.

Used by the analyst right after synthesize() -- every claim it just cited
gets reduced to typed facts and written to memory, which is what lets a
later question about the same entity retrieve them via
memory.get_facts_for_entity() instead of re-searching.

CHEAP_MODEL is deliberately used here: this is a narrow extraction task
(claim -> triples), not open-ended reasoning, and it runs once per claim
per question -- exactly the kind of high-volume, low-difficulty call the
two-tier routing in src/config.py exists for.
"""
from src.config import CHEAP_MODEL
from src.jsonutil import parse_json_response, JSONParseError
from src.llm import call_llm

_PROMPT = """Extract factual (subject, relation, object) triples from the claim below.

Rules:
- Only extract what the claim actually states. Do not infer or add anything.
- subject and object should be concrete entities or values (a company name, a person, a number, a date) -- not full sentences.
- relation should be a short verb phrase ("opened", "owns", "is owned by", "raised", "joined as", "was founded in").
- A single claim can yield 0, 1, or multiple triples. If the claim states nothing extractable as a clean triple, return an empty list.
- Return ONLY a JSON array, no other text. Example:
  [{{"subject": "Tanishq", "relation": "opened", "object": "120 new stores in the last year"}}]

Claim: {claim}
"""


def extract_facts_from_claim(claim: str, *, question_id: str) -> list[dict]:
    prompt = _PROMPT.format(claim=claim)
    result = call_llm(prompt, question_id=question_id, stage="extract_facts", model=CHEAP_MODEL)
    try:
        parsed = parse_json_response(result.text)
    except JSONParseError:
        return []
    if not isinstance(parsed, list):
        return []
    triples = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if {"subject", "relation", "object"} <= item.keys():
            triples.append(
                {
                    "subject": str(item["subject"]).strip(),
                    "relation": str(item["relation"]).strip(),
                    "object": str(item["object"]).strip(),
                }
            )
    return triples
