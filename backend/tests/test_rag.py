from app.config import DATA_DIR
from app.rag.store import load_policy_chunks


def test_policy_is_chunked_by_section_with_citation_metadata():
    chunks = load_policy_chunks(DATA_DIR)
    sections = [c.metadata["section"] for c in chunks]
    assert sections == ["Section 1: Home Water Damage Coverage", "Section 2: Personal Property Protection"]
    assert all(c.metadata["source"] == "sample_policy.md" for c in chunks)


def test_water_damage_query_retrieves_section_1(policy_store):
    top = policy_store.search("Is water damage from a burst pipe covered?", k=1)[0]
    assert top.section == "Section 1: Home Water Damage Coverage"
    assert "$25,000" in top.content
    assert top.citation == "sample_policy.md — Section 1: Home Water Damage Coverage"


def test_jewelry_query_retrieves_section_2(policy_store):
    top = policy_store.search("What is the limit for jewelry and electronics?", k=1)[0]
    assert top.section == "Section 2: Personal Property Protection"
    assert "$10,000" in top.content


def test_max_gap_drops_sections_trailing_the_top_hit(policy_store):
    query = "Is water damage from a burst pipe covered?"
    assert len(policy_store.search(query, k=3)) == 2  # unfiltered: both sections come back
    hits = policy_store.search(query, k=3, max_gap=0.0)
    assert [h.section for h in hits] == ["Section 1: Home Water Damage Coverage"]


def test_min_score_drops_off_topic_matches(policy_store):
    assert policy_store.search("What's the weather in Paris?", k=3, min_score=0.1) == []


def test_search_policy_tool_reports_no_match_when_everything_is_filtered(policy_store, claims_repo):
    from app.agent.tools import build_tools

    search = build_tools(policy_store, claims_repo, min_score=0.1)[0]
    msg = search.invoke({"type": "tool_call", "id": "1", "name": "search_policy", "args": {"query": "weather in Paris"}})
    assert msg.content == "No relevant policy sections found."
    assert msg.artifact == []


def test_ingest_is_idempotent(policy_store):
    assert policy_store.ingest(DATA_DIR) == 2
    assert len(policy_store.search("coverage", k=10)) == 2
