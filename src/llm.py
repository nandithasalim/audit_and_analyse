"""
Thin wrapper around the OpenAI Responses API that:
  1. makes the actual call (optionally with the web_search tool),
  2. extracts plain text + citation URLs from the response,
  3. logs token/dollar cost tagged by pipeline STAGE, so cost-per-question
     can later be broken down by stage rather than reported as one number.

Every other module (planner, retriever, synthesizer, auditor, repair)
should call `call_llm()` rather than touching the OpenAI client directly --
that's what makes the stage-level cost breakdown possible without
scattering bookkeeping through the whole codebase.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

from src.config import (
    OPENAI_API_KEY,
    PRICING_PER_1M_TOKENS,
    WEB_SEARCH_COST_PER_CALL_USD,
    USD_TO_INR,
    COST_LOG_PATH,
)

_client = OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class LLMResult:
    text: str
    citations: list = field(default_factory=list)   # list of {"url", "title"}
    input_tokens: int = 0
    output_tokens: int = 0
    search_calls: int = 0
    cost_usd: float = 0.0


def _extract_citations(response) -> list:
    """Pull {url, title} pairs out of a Responses API result that used web_search."""
    citations = []
    for item in getattr(response, "output", []) or []:
        # message items carry annotations with the actual citation URLs
        content = getattr(item, "content", None)
        if not content:
            continue
        for block in content:
            for ann in getattr(block, "annotations", []) or []:
                url = getattr(ann, "url", None)
                if url:
                    citations.append({"url": url, "title": getattr(ann, "title", "")})
    return citations


def _count_search_calls(response) -> int:
    return sum(
        1
        for item in getattr(response, "output", []) or []
        if getattr(item, "type", "") == "web_search_call"
    )


def _cost_usd(model: str, input_tokens: int, output_tokens: int, search_calls: int) -> float:
    rates = PRICING_PER_1M_TOKENS.get(model)
    if rates is None:
        raise ValueError(
            f"No pricing entry for model '{model}' -- add it to "
            f"PRICING_PER_1M_TOKENS in src/config.py"
        )
    token_cost = (input_tokens / 1_000_000) * rates["input"] + (
        output_tokens / 1_000_000
    ) * rates["output"]
    search_cost = search_calls * WEB_SEARCH_COST_PER_CALL_USD
    return token_cost + search_cost


def _log_cost(question_id: str, stage: str, model: str, result: LLMResult) -> None:
    Path(COST_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "question_id": question_id,
        "stage": stage,
        "model": model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "search_calls": result.search_calls,
        "cost_usd": round(result.cost_usd, 6),
        "cost_inr": round(result.cost_usd * USD_TO_INR, 4),
        "timestamp": time.time(),
    }
    with open(COST_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def call_llm(
    input_text: str,
    *,
    question_id: str,
    stage: str,
    model: str,
    use_web_search: bool = False,
    max_output_tokens: Optional[int] = None,
) -> LLMResult:
    """
    The one function every pipeline stage should call.

    question_id: which question this call belongs to (for cost-by-question)
    stage: "plan" | "retrieve" | "resolve_evidence" | "synthesize" |
           "audit" | "repair" -- whatever tag this call belongs to
           (for cost-by-stage)
    """
    tools = [{"type": "web_search"}] if use_web_search else None

    kwargs = {"model": model, "input": input_text}
    if tools:
        kwargs["tools"] = tools
    if max_output_tokens:
        kwargs["max_output_tokens"] = max_output_tokens

    response = _client.responses.create(**kwargs)

    result = LLMResult(
        text=response.output_text,
        citations=_extract_citations(response),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        search_calls=_count_search_calls(response),
    )
    result.cost_usd = _cost_usd(
        model, result.input_tokens, result.output_tokens, result.search_calls
    )
    _log_cost(question_id, stage, model, result)
    return result
