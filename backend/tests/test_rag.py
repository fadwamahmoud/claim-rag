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


def test_ingest_is_idempotent(policy_store):
    assert policy_store.ingest(DATA_DIR) == 2
    assert len(policy_store.search("coverage", k=10)) == 2
