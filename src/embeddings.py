"""
OpenAI embeddings wrapper for real semantic similarity, used by
src/retriever.py instead of TF-IDF's bag-of-words overlap.

Why this over a local model (sentence-transformers): cost is negligible
next to the chat models ($0.02/1M tokens vs $2-50/1M -- see
PRICING_PER_1M_TOKENS in config.py) and it avoids adding a multi-hundred-MB
torch dependency for a take-home on a deadline. That's a documented
trade-off, not a "best" choice -- a fully local/free alternative is listed
under DECISIONS.md's "what's next".

NOTE: this cannot be exercised in an offline test (unlike retriever.py's
TF-IDF path) -- it needs a real API key and real network. See
tests/test_offline.py for what's covered without one, and
scripts/inspect_retrieval.py --semantic for a way to sanity-check this
against the same worked Tanishq/Titan example used for the TF-IDF failure
case, once you have a live key.
"""
import json
import math
import time
from pathlib import Path

from openai import OpenAI

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY, PRICING_PER_1M_TOKENS, COST_LOG_PATH, USD_TO_INR

_client = OpenAI(api_key=OPENAI_API_KEY)

# In-process cache only (not persisted) -- keyed by (model, text). Cuts
# duplicate embedding calls within a single pipeline run (e.g. the same
# claim text embedded once per audit rather than re-embedded if referenced
# twice), not across separate `python scripts/run_pipeline.py` invocations.
_cache: dict[tuple, list[float]] = {}

# OpenAI enforces a hard per-request token cap on embeddings (300k as of
# writing). A single large cited page can chunk into far more text than
# that in one go -- this bit in practice on q08, whose cited page (Titan's
# official disclosures) chunked into >1.6M tokens' worth of paragraphs in
# a single select_relevant_chunks() call, which handed all of them to
# embed() at once.
#
# First attempt at batching used a flat 4-chars-per-token guess with no
# tokenizer dependency -- that undercounted badly on this content (dense
# Indian-business text: currency symbols, numbers, company names) and a
# "safely under budget" batch still blew the real cap on retry (est.
# 250k tokens, actual 467,946). Using tiktoken's real cl100k_base
# tokenizer instead (the encoding text-embedding-3-small actually uses)
# fixes the root cause -- exact counts, not a guess. Falls back to a much
# more conservative character estimate only if tiktoken's encoding file
# can't be loaded (e.g. no network for its one-time download), since a
# wrong guess is what caused this bug in the first place.
try:
    import tiktoken
    _encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoding = None

_MAX_TOKENS_PER_REQUEST = 280_000  # real cap is 300k; margin is for safety, not for estimate error, when using real counts
_MAX_TOKENS_PER_REQUEST_FALLBACK = 50_000  # much smaller budget when we're only guessing at token count
_CHARS_PER_TOKEN_FALLBACK_ESTIMATE = 2  # deliberately pessimistic given what the 4:1 guess got wrong here

# OpenAI also caps a single item within a request at 8192 tokens on its own
# (separate from the whole-request cap above) -- q08 hit this too, from a
# chunk_text() bug now fixed in src/fetch.py (a data-table-like block with
# no sentence punctuation came out as one giant unsplit chunk). That's the
# real fix; this is a second line of defense so any other caller handing
# embed() an unusually long single string clips instead of crashing the
# whole run three questions deep into a paid batch.
_MAX_TOKENS_PER_ITEM = 8000  # real cap is 8192; small margin


def _count_tokens(text: str) -> int:
    if _encoding is not None:
        return len(_encoding.encode(text))
    return max(1, len(text) // _CHARS_PER_TOKEN_FALLBACK_ESTIMATE)


def _clip_to_item_limit(text: str) -> str:
    """Truncates a single text so it can't exceed the per-item token cap.
    Silent truncation is an acceptable last resort here (not the primary
    fix) -- it only ever fires if something upstream produced a chunk this
    large in the first place, which the fetch.py fix above should prevent
    for the auditor's own path."""
    if _encoding is not None:
        tokens = _encoding.encode(text)
        if len(tokens) <= _MAX_TOKENS_PER_ITEM:
            return text
        return _encoding.decode(tokens[:_MAX_TOKENS_PER_ITEM])
    max_chars = _MAX_TOKENS_PER_ITEM * _CHARS_PER_TOKEN_FALLBACK_ESTIMATE
    return text[:max_chars]


def _log_cost(question_id: str, stage: str, n_tokens: int) -> None:
    rate = PRICING_PER_1M_TOKENS.get(EMBEDDING_MODEL, {}).get("input", 0.0)
    cost_usd = (n_tokens / 1_000_000) * rate
    Path(COST_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "question_id": question_id, "stage": stage, "model": EMBEDDING_MODEL,
        "input_tokens": n_tokens, "output_tokens": 0, "search_calls": 0,
        "cost_usd": round(cost_usd, 6), "cost_inr": round(cost_usd * USD_TO_INR, 4),
        "timestamp": time.time(),
    }
    with open(COST_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _batch_originals_by_token_budget(originals: list[str]) -> list[list[str]]:
    """Groups the ORIGINAL texts into request-sized batches, sized by the
    token count of what will actually be sent (the clipped version), so a
    batch never exceeds the per-request budget even after per-item clipping."""
    budget = _MAX_TOKENS_PER_REQUEST if _encoding is not None else _MAX_TOKENS_PER_REQUEST_FALLBACK
    batches: list[list[str]] = []
    batch: list[str] = []
    batch_tokens = 0
    for orig in originals:
        t_tokens = _count_tokens(_clip_to_item_limit(orig))
        if batch and batch_tokens + t_tokens > budget:
            batches.append(batch)
            batch, batch_tokens = [], 0
        batch.append(orig)
        batch_tokens += t_tokens
    if batch:
        batches.append(batch)
    return batches


def embed(texts: list[str], *, question_id: str = "unknown", stage: str = "embed") -> list[list[float]]:
    """Embeds a batch of texts, using and populating the in-process cache.
    Returns one embedding vector per input text, same order. Cached and
    returned under the ORIGINAL text even where the sent text was clipped,
    so callers matching chunks back to their original text still work."""
    if not texts:
        return []

    to_fetch = [t for t in texts if (EMBEDDING_MODEL, t) not in _cache]
    for group in _batch_originals_by_token_budget(to_fetch):
        send_group = [_clip_to_item_limit(t) for t in group]
        response = _client.embeddings.create(model=EMBEDDING_MODEL, input=send_group)
        for orig_text, item in zip(group, response.data):
            _cache[(EMBEDDING_MODEL, orig_text)] = item.embedding
        if hasattr(response, "usage") and response.usage:
            _log_cost(question_id, stage, response.usage.total_tokens)

    return [_cache[(EMBEDDING_MODEL, t)] for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
