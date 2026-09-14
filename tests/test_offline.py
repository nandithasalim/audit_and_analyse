"""
Offline tests -- no network, no OpenAI key needed. Run these from a clean
checkout to sanity-check the parts of the pipeline that don't depend on
live web access:

    python tests/test_offline.py

This exists because the cloud sandbox this project was *built* in has no
outbound network access at all (see DECISIONS.md) -- these are the tests
that could actually be run and verified there. Anything that needs
call_llm() (plan/retrieve/synthesize/audit/repair) needs a real
OPENAI_API_KEY and real internet and can only be verified on a normal
machine -- see scripts/run_pipeline.py for that.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config

# point the memory store at a scratch DB before importing src.memory's
# module-level DB_PATH reference
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
config.DB_PATH = _tmp_db.name

from src import memory  # noqa: E402
from src.fetch import Chunk, chunk_text  # noqa: E402
from src.jsonutil import parse_json_response, JSONParseError  # noqa: E402
from src.retriever import select_relevant_chunks  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def test_jsonutil():
    print("\n[jsonutil]")
    check("plain json", parse_json_response('{"a": 1}') == {"a": 1})
    check("fenced json", parse_json_response('```json\n{"a": 1}\n```') == {"a": 1})
    check("json with preamble", parse_json_response('Here you go:\n{"a": 1}\nHope that helps!') == {"a": 1})
    check("array", parse_json_response('[1, 2, 3]') == [1, 2, 3])
    try:
        parse_json_response("no json here at all")
        check("raises on garbage", False)
    except JSONParseError:
        check("raises on garbage", True)


def test_chunk_text():
    print("\n[fetch.chunk_text]")
    text = "Short.\n\n" + ("This is a real paragraph with enough content to survive the min-length filter. " * 2) + \
           "\n\n" + ("A very long run-on paragraph. " * 60)
    chunks = chunk_text(text, url="http://example.com")
    check("drops too-short fragments", all(len(c.text) >= 40 for c in chunks))
    check("splits overlong paragraphs", all(len(c.text) <= 1200 for c in chunks), f"max len seen: {max((len(c.text) for c in chunks), default=0)}")
    check("produced at least one chunk", len(chunks) > 0)


def test_memory_hybrid_retrieval():
    print("\n[memory: hybrid FTS + TF-IDF retrieval]")
    memory.init_db()
    memory.save_fact("Tanishq", "opened", "120 new stores in the last year", question_id="t1", source_url="http://example.com/a")
    memory.save_fact("Titan Company", "owns", "Tanishq", question_id="t1", source_url="http://example.com/b")
    memory.save_fact("Zepto", "raised", "$350M Series F", question_id="t2", source_url="http://example.com/c")

    hits = memory.get_facts_for_entity("Tanishq")
    subjects_or_objects = {(h["subject"], h["object"]) for h in hits}
    check("exact-token entity match found via FTS", any("Tanishq" in s or "Tanishq" in o for s, o in subjects_or_objects))
    check("unrelated entity not returned", not any(h["subject"] == "Zepto" for h in hits))

    fuzzy_hits = memory.get_facts_for_entity("Tanishq stores openings")
    check("TF-IDF fallback finds partial-overlap fact", any("stores" in h["object"] for h in fuzzy_hits), str(fuzzy_hits))


def test_retriever_known_limitation():
    """
    Reproduces, with realistic synthetic Wikipedia-style paragraphs (no
    network available in this sandbox -- see DECISIONS.md), the retrieval
    failure mode documented in src/retriever.py: a TF-IDF ranker can rate a
    chunk that repeats the claim's keywords without stating the
    relationship above the one chunk that actually states it, when that
    chunk phrases the relationship differently than the claim does.
    """
    print("\n[retriever: known limitation, reproduced offline]")
    claim = "Tanishq is owned by Titan Company"

    chunks = [
        Chunk(
            index=0, url="http://example.com/wiki",
            text=("In fiscal year 2025, Titan Company's Tanishq division opened approximately 120 "
                  "new stores across India, expanding into tier-2 and tier-3 cities as part of its "
                  "continued retail growth strategy under Titan Company's broader expansion plan."),
        ),
        Chunk(
            index=1, url="http://example.com/wiki",
            text=("Titan Company Limited v. State of Karnataka was a tax dispute heard in 2019 "
                  "concerning import duties on gold used by the Tanishq brand, decided in favour of "
                  "Titan Company after Tanishq's legal team argued the classification was incorrect."),
        ),
        Chunk(
            index=2, url="http://example.com/wiki",
            text=("Tanishq is an Indian jewellery and watch brand, a subsidiary of the Tata Group's "
                  "Titan Company, headquartered in Bangalore and launched in 1994."),
        ),
        Chunk(
            index=3, url="http://example.com/wiki",
            text=("Tanishq has since grown into India's largest jewellery retailer by revenue, "
                  "competing with Kalyan Jewellers and Malabar Gold and Diamonds in a market worth "
                  "billions of dollars annually."),
        ),
    ]

    ranked = select_relevant_chunks(claim, chunks, top_k=len(chunks))
    print("  Full ranking for claim:", claim)
    for rank, r in enumerate(ranked, start=1):
        print(f"    #{rank} score={r['score']:.4f} chunk#{r['index']}: {r['text'][:90]}...")

    top_pick_index = ranked[0]["index"]
    ownership_chunk_rank = next(i for i, r in enumerate(ranked, start=1) if r["index"] == 2)
    print(f"  -> The chunk that actually states ownership (chunk#2) ranked #{ownership_chunk_rank} of {len(ranked)}.")
    print(f"  -> The top-ranked chunk was chunk#{top_pick_index}"
          f" ({'the ownership chunk -- ranker got it right here' if top_pick_index == 2 else 'NOT the ownership chunk -- this is the documented failure mode'}).")
    # Not asserted as a pass/fail check -- the point of this test is to observe and report
    # the actual ranking, which is what DECISIONS.md's limitation section is built from.


if __name__ == "__main__":
    test_jsonutil()
    test_chunk_text()
    test_memory_hybrid_retrieval()
    test_retriever_known_limitation()

    print(f"\n{PASS} passed, {FAIL} failed")
    Path(_tmp_db.name).unlink(missing_ok=True)
    sys.exit(1 if FAIL else 0)
