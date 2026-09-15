"""
Lightweight source-trust ranking, used to help the analyst pick a side when
sources genuinely disagree (the brief's own bonus, and what q07 -- the
Blinkit funding round question -- is built to exercise) instead of
reporting both numbers and shrugging.

This is a heuristic domain allowlist, not a real credibility model, and
that's a deliberate trade-off: a proper source-trust system would need a
maintained, sourced reputation database, which is out of scope for a
take-home. What this gives instead is honest and narrow -- a domain not on
the list comes back "unranked" rather than a false confidence signal, and
the tier is one input to the synthesize prompt's reasoning, not an
automatic override (a very recent unranked source can still beat a stale
tier-2 one; the LLM is told to weigh both, not just sort by tier).
"""
import re
from urllib.parse import urlparse

# Tier 2: established, edited news/business publications -- not infallible,
# but have editorial process and correction norms.
TIER_2_ESTABLISHED_NEWS = {
    "economictimes.indiatimes.com", "bsmedia.business-standard.com", "business-standard.com",
    "livemint.com", "moneycontrol.com", "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
    "thehindubusinessline.com", "financialexpress.com", "cnbctv18.com", "ndtv.com",
    "hindustantimes.com", "timesofindia.indiatimes.com", "en.wikipedia.org",
}

# Tier 3: trade/industry press and startup-focused outlets -- useful for
# funding/startup-specific coverage (their whole beat), but smaller
# editorial teams and more reliant on press releases than tier 2.
TIER_3_TRADE_PRESS = {
    "entrackr.com", "inc42.com", "yourstory.com", "techcrunch.com", "vccircle.com",
    "techinasia.com", "themorningcontext.com",
}

# Regulatory/exchange filing sources -- as authoritative as it gets for a
# specific disclosed fact, though narrower in what they cover.
TIER_1_REGULATORY = {
    "nseindia.com", "bseindia.com", "sebi.gov.in", "mca.gov.in", "zaubacorp.com",
}


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _looks_like_own_domain(url: str, entity_hint: str) -> bool:
    """Heuristic only: does the entity's name (stripped of generic suffixes
    like "Company"/"Limited") show up in the domain? Catches Titan's own
    titancompany.in for "Titan Company" without a maintained per-entity
    domain list, at the cost of false negatives for a company whose domain
    doesn't resemble its name (and occasional false positives for an
    unrelated domain that happens to share a word)."""
    if not entity_hint:
        return False
    domain = _domain(url)
    if not domain:
        return False
    words = re.findall(r"[a-z0-9]+", entity_hint.lower())
    stopwords = {"company", "limited", "ltd", "inc", "the", "group", "india", "pvt", "private"}
    significant = [w for w in words if w not in stopwords and len(w) > 2]
    return any(w in domain for w in significant)


def trust_tier(url: str, entity_hint: str = "") -> tuple:
    """Returns (tier: int, label: str). Lower tier number = more trusted.
    tier 4 ("unranked") is the honest default for anything not recognized."""
    domain = _domain(url)
    if not domain:
        return (4, "unranked")
    if domain in TIER_1_REGULATORY or _looks_like_own_domain(url, entity_hint):
        return (1, "official/regulatory source")
    if domain in TIER_2_ESTABLISHED_NEWS:
        return (2, "established news")
    if domain in TIER_3_TRADE_PRESS:
        return (3, "trade/industry press")
    return (4, "unranked")


def tag_citations(citations: list[dict], entity_hint: str = "") -> str:
    """Human-readable trust summary for a list of {"url","title"} citations,
    to embed in the evidence string handed to synthesize()."""
    if not citations:
        return "no citations"
    tagged = []
    for c in citations:
        tier, label = trust_tier(c.get("url", ""), entity_hint)
        tagged.append(f"{c.get('title','') or c['url']} [tier {tier}: {label}]")
    return "; ".join(tagged)
