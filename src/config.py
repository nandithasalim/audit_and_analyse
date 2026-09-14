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
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "gpt-5.6-luna")

USD_TO_INR = float(os.getenv("USD_TO_INR", "88.0"))

# $ per 1M tokens. VERIFY AND UPDATE against the live pricing page before
# reporting numbers in DECISIONS.md -- these are placeholders to get you
# running, not something to cite as-is.
PRICING_PER_1M_TOKENS = {
    "gpt-6-astra": {"input": 1.00, "output": 6.00},
    "gpt-5.6-luna": {"input": 0.10, "output": 0.60},
}

# web_search tool: $10 per 1,000 calls, plus the retrieved content is
# billed as normal input tokens on whichever model made the call.
WEB_SEARCH_COST_PER_CALL_USD = 10.0 / 1000

# Where things get written
DB_PATH = "data/memory.db"
TRACES_DIR = "traces"
COST_LOG_PATH = "traces/costs.jsonl"
