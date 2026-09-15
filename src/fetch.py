"""
Independent page fetching for the auditor.

This is deliberately separate from the analyst's retrieval path. The analyst
gets page content indirectly, mediated by OpenAI's web_search tool -- it
never sees raw HTML, only whatever the model chose to summarize. If the
auditor reused that same path, it would be re-asking the same model to
grade its own homework using its own summary of the evidence -- not an
independent check. So the auditor fetches the cited URL itself, with plain
httpx + BeautifulSoup, and reasons over the actual page text.

Trade-off: this is slower and dumber than a real extraction pipeline
(trafilatura/readability), and it will do badly on JS-rendered pages or
paywalls. Documented in DECISIONS.md rather than solved -- a take-home
budget doesn't stretch to a full readability implementation, and a naive
paragraph extractor is enough to demonstrate the independent-verification
architecture.
"""
import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; DylaAuditorBot/0.1; +research-agent-take-home)"


@dataclass
class Chunk:
    text: str
    index: int
    url: str


def fetch_url(url: str, *, timeout: float = 15.0) -> str:
    """Fetch a URL and return extracted, cleaned body text. Raises on failure --
    callers (the auditor) must treat a fetch failure as its own audit outcome
    ("could not verify"), not silently skip the claim."""
    resp = httpx.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    # Prefer <article>/<main> when the page has one -- cuts down on nav/boilerplate
    # text competing for TF-IDF weight against the actual content.
    container = soup.find("article") or soup.find("main") or soup.body or soup
    text = container.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, url: str, *, min_len: int = 40, max_len: int = 1200) -> list[Chunk]:
    """
    Paragraph-level chunking: split on blank lines, drop boilerplate-sized
    fragments (nav labels, single dates, etc.), and hard-split anything
    implausibly long (a whole unbroken article dumped as one "paragraph",
    which happens on sites that don't use <p> tags cleanly).
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[Chunk] = []
    idx = 0

    def _emit(piece: str) -> None:
        nonlocal idx
        piece = piece.strip()
        if piece:
            chunks.append(Chunk(text=piece, index=idx, url=url))
            idx += 1

    for para in raw_paragraphs:
        if len(para) < min_len:
            continue
        if len(para) <= max_len:
            _emit(para)
            continue

        # split long paragraphs on sentence boundaries into ~max_len pieces
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for s in sentences:
            # A single "sentence" can itself be longer than max_len -- a
            # data table or a dense disclosure block with no sentence-
            # ending punctuation, which the split above never breaks up.
            # This is exactly what broke q08's audit: the previous version
            # of this function emitted that one oversized "sentence" as a
            # single chunk (contradicting this function's own docstring,
            # which already claimed everything gets hard-split), and it
            # later blew OpenAI's per-item embedding token limit. Hard-split
            # anything this long on its own, character-wise, instead.
            if len(s) > max_len:
                if buf:
                    _emit(buf)
                    buf = ""
                for i in range(0, len(s), max_len):
                    _emit(s[i:i + max_len])
                continue
            if len(buf) + len(s) + 1 > max_len and buf:
                _emit(buf)
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            _emit(buf)
    return chunks
