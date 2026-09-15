"""
Evidence-chunk selection: given a claim (or a question) and a list of
Chunks, rank them by relevance and return the top few as "evidence" for
the auditor / analyst to reason over.

select_relevant_chunks() is now embedding-based (semantic similarity via
src/embeddings.py) -- this replaces the original TF-IDF ranker, which is
kept below as select_relevant_chunks_tfidf() purely as the documented
before/after: TF-IDF scores by vocabulary overlap, not by whether a chunk
actually states the relationship being asked about, which is exactly why
"unsupported" from the old pipeline could mean "the ranker picked the
wrong paragraph" as often as "the source doesn't say this" -- see
DECISIONS.md and tests/test_offline.py::test_retriever_known_limitation
for the worked Tanishq/Titan case this was built to fix. Embeddings aren't
a silver bullet either (they can still miss a fact stated in an unusual
way, or over-match on topical similarity without the specific relationship)
-- there's no offline test for the embedding path since it needs a real
API key and network; sanity-check it yourself with
scripts/inspect_retrieval.py against the same example before trusting it
in DECISIONS.md.
"""
from src.embeddings import cosine_similarity, embed
from src.fetch import Chunk


def select_relevant_chunks(claim: str, chunks: list[Chunk], top_k: int = 5, *, question_id: str = "unknown") -> list[dict]:
    """Returns up to top_k chunks as [{"text", "index", "url", "score"}, ...],
    ranked by semantic (embedding cosine) similarity to `claim`, highest first."""
    if not chunks:
        return []

    texts = [c.text for c in chunks]
    embeddings = embed(texts + [claim], question_id=question_id, stage="embed")
    chunk_embeddings, claim_embedding = embeddings[:-1], embeddings[-1]

    sims = [cosine_similarity(ce, claim_embedding) for ce in chunk_embeddings]
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    return [
        {"text": c.text, "index": c.index, "url": c.url, "score": float(score)}
        for c, score in ranked[:top_k]
    ]


def rank_all_chunks(claim: str, chunks: list[Chunk], *, question_id: str = "unknown") -> list[dict]:
    """Same as select_relevant_chunks but returns every chunk ranked, not just
    the top_k -- used by scripts/inspect_retrieval.py to see where the
    correct paragraph actually landed when the top-k pick was wrong."""
    return select_relevant_chunks(claim, chunks, top_k=len(chunks), question_id=question_id)


def select_relevant_chunks_tfidf(claim: str, chunks: list[Chunk], top_k: int = 5) -> list[dict]:
    """The original TF-IDF ranker. No longer used in the live pipeline --
    kept for the offline test/documented comparison (see module docstring).
    Fully local, no API key or network needed, which is *why* it was the
    original choice before an embedding path was worth the added cost/
    dependency."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

    if not chunks:
        return []
    corpus = [c.text for c in chunks]
    try:
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform(corpus + [claim])
        sims = sk_cosine_similarity(matrix[-1], matrix[:-1])[0]
    except ValueError:
        return []
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    return [
        {"text": c.text, "index": c.index, "url": c.url, "score": float(score)}
        for c, score in ranked[:top_k]
    ]
