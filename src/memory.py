"""
Memory store: a typed relationship graph (subject, relation, object triples,
each tagged with the question that produced it and the source URL that backs
it) plus hybrid lexical + TF-IDF retrieval over it.

Why a fact graph and not "save the whole answer text": the brief's hard
bonus is memory that *transfers* to a question you haven't seen, not caching
an answer you've already seen. Storing atomic typed facts (e.g.
("Tanishq", "opened", "120 new stores in the last year")) means a later
question about a *different* aspect of the same entity ("who owns Tanishq")
can still be answered from facts already on file, or at minimum lets the
analyst skip straight to a narrower follow-up search instead of re-deriving
everything about the entity from scratch.

Retrieval is hybrid because the two signals fail differently:
  - SQLite FTS5 (bm25-ranked) is fast and precise on exact/near-exact token
    overlap ("Tanishq" matches "Tanishq") but is blind to a fact stored under
    a differently-worded relation than the new question uses.
  - A TF-IDF cosine pass over the same corpus catches partial/fuzzy overlap
    FTS5's tokenizer misses, at the cost of being noisier.
Scores from both are min-max normalized per query and blended.

NOTE: this is the same family of ranker used in src/retriever.py for
evidence-chunk selection, and it has the same known failure mode -- see
DECISIONS.md "Where it breaks" for the worked Tanishq/Titan example where a
TF-IDF-style ranker over-weights shared surface words and can miss the one
sentence that actually states a relationship phrased differently than the
query.
"""
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT UNIQUE NOT NULL,
    title       TEXT,
    fetched_at  REAL NOT NULL,
    raw_text    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    subject      TEXT NOT NULL,
    relation     TEXT NOT NULL,
    object       TEXT NOT NULL,
    source_url   TEXT,
    question_id  TEXT NOT NULL,
    confidence   REAL NOT NULL DEFAULT 1.0,
    created_at   REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    subject, relation, object,
    content='facts', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, subject, relation, object)
    VALUES (new.id, new.subject, new.relation, new.object);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, subject, relation, object)
    VALUES ('delete', old.id, old.subject, old.relation, old.object);
END;
"""


@contextmanager
def _conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(_SCHEMA)


def save_document(url: str, title: str, raw_text: str) -> int:
    """Idempotent: re-fetching the same URL updates it in place."""
    with _conn() as conn:
        conn.execute(
            """INSERT INTO documents (url, title, fetched_at, raw_text)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(url) DO UPDATE SET
                 title=excluded.title, fetched_at=excluded.fetched_at, raw_text=excluded.raw_text""",
            (url, title, time.time(), raw_text),
        )
        row = conn.execute("SELECT id FROM documents WHERE url = ?", (url,)).fetchone()
        return row["id"]


def get_document(url: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE url = ?", (url,)).fetchone()
        return dict(row) if row else None


def save_fact(
    subject: str,
    relation: str,
    obj: str,
    *,
    question_id: str,
    source_url: Optional[str] = None,
    confidence: float = 1.0,
) -> int:
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO facts (subject, relation, object, source_url, question_id, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (subject.strip(), relation.strip(), obj.strip(), source_url, question_id, confidence, time.time()),
        )
        return cur.lastrowid


def save_facts(triples: list[dict], *, question_id: str) -> int:
    """triples: [{"subject":..., "relation":..., "object":..., "source_url":..., "confidence":...}, ...]"""
    n = 0
    for t in triples:
        if not t.get("subject") or not t.get("relation") or not t.get("object"):
            continue
        save_fact(
            t["subject"], t["relation"], t["object"],
            question_id=question_id,
            source_url=t.get("source_url"),
            confidence=t.get("confidence", 1.0),
        )
        n += 1
    return n


def _fts_search(entity: str, top_k: int) -> list[dict]:
    # FTS5 MATCH needs its query tokens escaped/quoted to tolerate punctuation
    # in entity names (e.g. "Tata Group's").
    tokens = re.findall(r"[A-Za-z0-9]+", entity)
    if not tokens:
        return []
    query = " OR ".join(f'"{t}"' for t in tokens)
    with _conn() as conn:
        rows = conn.execute(
            """SELECT f.*, bm25(facts_fts) AS rank
               FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid
               WHERE facts_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (query, top_k * 3),
        ).fetchall()
    out = [dict(r) for r in rows]
    # bm25() in SQLite is *lower is better*; flip sign so higher = more relevant,
    # matching the convention the TF-IDF pass below uses.
    for r in out:
        r["_fts_score"] = -r.pop("rank")
    return out


def _tfidf_search(entity: str, top_k: int) -> list[dict]:
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM facts").fetchall()]
    if not rows:
        return []
    corpus = [f"{r['subject']} {r['relation']} {r['object']}" for r in rows]
    try:
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform(corpus + [entity])
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
    except ValueError:
        # empty vocabulary (e.g. entity is only stopwords/punctuation)
        return []
    ranked = sorted(zip(rows, sims), key=lambda x: x[1], reverse=True)[:top_k]
    out = []
    for r, score in ranked:
        if score <= 0:
            continue
        r = dict(r)
        r["_tfidf_score"] = float(score)
        out.append(r)
    return out


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def get_facts_for_entity(entity: str, top_k: int = 10) -> list[dict]:
    """
    Hybrid retrieval: blend FTS5 (bm25) and TF-IDF cosine results, each
    min-max normalized to [0, 1] within its own result set, weighted 50/50,
    then merged by fact id (max of the two normalized scores wins when a
    fact shows up in both lists).
    """
    fts_hits = _fts_search(entity, top_k)
    tfidf_hits = _tfidf_search(entity, top_k)

    fts_scores = _normalize([h["_fts_score"] for h in fts_hits])
    tfidf_scores = _normalize([h["_tfidf_score"] for h in tfidf_hits])

    blended: dict[int, dict] = {}
    for h, s in zip(fts_hits, fts_scores):
        blended[h["id"]] = {**h, "_blend_score": 0.5 * s}
    for h, s in zip(tfidf_hits, tfidf_scores):
        if h["id"] in blended:
            blended[h["id"]]["_blend_score"] += 0.5 * s
        else:
            blended[h["id"]] = {**h, "_blend_score": 0.5 * s}

    ranked = sorted(blended.values(), key=lambda r: r["_blend_score"], reverse=True)
    return ranked[:top_k]


def all_facts(question_id: Optional[str] = None) -> list[dict]:
    with _conn() as conn:
        if question_id:
            rows = conn.execute("SELECT * FROM facts WHERE question_id = ?", (question_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM facts").fetchall()
        return [dict(r) for r in rows]
