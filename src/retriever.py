"""
TF-IDF chunk selection: given a claim and a list of Chunks fetched from the
cited page, rank chunks by cosine similarity and return the top few as
"evidence" for the auditor to reason over.

KNOWN LIMITATION (see DECISIONS.md "Where it breaks" for the worked
example): this is a bag-of-words ranker. It scores a chunk by vocabulary
overlap with the claim, not by whether the chunk actually states the
relationship the claim asserts. A claim like "Tanishq is owned by Titan
Company" and a page sentence like "Tanishq is a brand of Titan Company, part
of the Tata Group" overlap enough to probably rank -- but a differently
phrased statement of the same fact elsewhere on the page can lose to
chunks that just happen to repeat "Titan" and "Tanishq" many times without
stating the relationship at all (a legal-case caption, a filings directory
listing, a store-count news blurb). This is exactly the gap an embedding
retriever or a query-expansion step would close, and exactly why the
auditor's verdict is only as trustworthy as the chunks handed to it --
"unsupported" from this pipeline can mean "the fact is false" or it can
mean "the ranker didn't surface the right paragraph." The auditor's output
schema keeps these distinguishable by requiring the reasoning field to
quote what it saw, so a human reviewing the trace can tell the difference;
the pipeline itself cannot.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.fetch import Chunk


def select_relevant_chunks(claim: str, chunks: list[Chunk], top_k: int = 5) -> list[dict]:
    """Returns up to top_k chunks as [{"text", "index", "url", "score"}, ...],
    ranked highest score first. Empty list if there's nothing to rank against."""
    if not chunks:
        return []

    corpus = [c.text for c in chunks]
    try:
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform(corpus + [claim])
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
    except ValueError:
        # e.g. every chunk + the claim reduce to an empty vocabulary
        return []

    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    return [
        {"text": c.text, "index": c.index, "url": c.url, "score": float(score)}
        for c, score in ranked[:top_k]
    ]


def rank_all_chunks(claim: str, chunks: list[Chunk]) -> list[dict]:
    """Same as select_relevant_chunks but returns every chunk ranked, not just
    the top_k -- used by scripts/inspect_retrieval.py to see where the
    correct paragraph actually landed when the top-k pick was wrong."""
    return select_relevant_chunks(claim, chunks, top_k=len(chunks))
