"""
Central config: loads .env, defines which model is used for which job,
and holds the pricing table the cost tracker needs.

IMPORTANT: model names and prices below were correct as of when this was
written but OpenAI changes both over time -- before you rely on the cost
numbers in your write-up, double check current prices at
https://developers.openai.com/api/docs/pricing and current available
models for your account, and update PRICING_PER_1M_TOKENS below.
"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
    )

# Two-tier model routing: STRONG for planning/synthesis (needs real reasoning),
# CHEAP for narrow yes/no judgments (agreement gate, relevance checks, etc.)
# -- this is the cost-aware routing lever from the design discussion.
STRONG_MODEL = os.getenv("STRONG_MODEL", "gpt-6-astra")
# NOTE: "gpt-5.5-mini" (the original placeholder here) does not exist on a
# real account -- confirmed via `python scripts/list_models.py` against a
# live key. The actual budget-tier model is "gpt-5.6-luna".
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "gpt-5.6-luna")

# retrieve() and repair's re-search fire N web_search-enabled calls per
# question and are the highest-volume, most expensive step in the whole
# pipeline -- but "search and report back what you found" doesn't need
# flagship-tier reasoning the way plan() and synthesize() do. Web search
# works on all model tiers per OpenAI's pricing page, so this is a real
# lever: gpt-5.6-terra is 5x/4.2x cheaper than astra on input/output.
# Worth comparing answer quality against STRONG_MODEL on a couple of
# questions before trusting this at face value in DECISIONS.md.
SEARCH_MODEL = os.getenv("SEARCH_MODEL", "gpt-5.6-terra")

# synthesize() doesn't call web_search itself -- it reasons over evidence
# already retrieved -- but it's not a narrow classification task like
# audit_claim either: it's where trust-tier disagreement resolution
# actually happens, and its output is the literal cited, user-facing
# answer. A step down from full STRONG_MODEL, not all the way to CHEAP:
# gpt-5.6-sol is half of astra's price on both input and output. Not
# verified against astra for quality on the trust-tier reasoning
# specifically -- compare a couple of answers before trusting this at
# face value in DECISIONS.md.
SYNTH_MODEL = os.getenv("SYNTH_MODEL", "gpt-5.6-sol")

# Used for semantic (embedding-based) similarity in src/retriever.py and
# src/memory.py, instead of TF-IDF's bag-of-words overlap. Cost is
# negligible next to the chat models above ($0.02/1M tokens vs $10-50/1M) --
# not the lever to pull if you need to cut cost further, that's still
# STRONG_MODEL usage in plan/retrieve/synthesize.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

USD_TO_INR = float(os.getenv("USD_TO_INR", "88.0"))

# $ per 1M tokens. VERIFY AND UPDATE against the live pricing page before
# reporting numbers in DECISIONS.md -- these are placeholders to get you
# running, not something to cite as-is.
PRICING_PER_1M_TOKENS = {
    # Real rates as of Sep 2026 (checked against public pricing coverage --
    # verify again yourself before citing these in DECISIONS.md, same
    # warning as always). The original placeholders here were ~5-10x too
    # low, which means every cost_usd number printed by the pipeline before
    # this fix understated the real spend -- your actual OpenAI bill was
    # never wrong (that's computed by OpenAI's own servers, not this file),
    # only what this code *reported* to you was.
    "gpt-6-astra": {"input": 10.00, "output": 50.00},
    "gpt-5.6-sol": {"input": 4.00, "output": 20.00},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},  # embeddings have no output tokens
}

# web_search tool: $10 per 1,000 calls, plus the retrieved content is
# billed as normal input tokens on whichever model made the call.
WEB_SEARCH_COST_PER_CALL_USD = 10.0 / 1000

# Where things get written
DB_PATH = "data/memory.db"
TRACES_DIR = "traces"
COST_LOG_PATH = "traces/costs.jsonl"
