"""
Small helper for the "ask the model for JSON, get back a string" pattern
used throughout the analyst and auditor. We don't rely on a structured
output / JSON-mode API parameter here (kept call_llm's surface minimal,
see src/llm.py) -- instead every prompt that wants JSON back is told to
emit ONLY a JSON object/array, and this strips the common ways models
violate that (markdown code fences, a leading "Here is the JSON:" sentence)
before parsing.
"""
import json
import re


class JSONParseError(ValueError):
    def __init__(self, raw_text: str, original: Exception):
        self.raw_text = raw_text
        self.original = original
        super().__init__(f"Could not parse JSON from model output: {original}\n--- raw ---\n{raw_text[:500]}")


def parse_json_response(text: str):
    """Best-effort extraction of a JSON value from an LLM text response."""
    candidate = text.strip()

    # strip ```json ... ``` or ``` ... ``` fences
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # fall back to grabbing the first balanced {...} or [...] block
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = candidate.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(candidate)):
            if candidate[i] == open_ch:
                depth += 1
            elif candidate[i] == close_ch:
                depth -= 1
                if depth == 0:
                    snippet = candidate[start : i + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError as e:
                        last_err = e
                    break
    raise JSONParseError(text, ValueError("no parseable JSON object/array found"))
