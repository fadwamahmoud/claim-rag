import json

import pytest
from pydantic import ValidationError

from app.agent.tools import build_tools
from app.tools.claims import ClaimNotFoundError, ClaimSubmission
from tests.conftest import read_claims


@pytest.fixture
def tools(policy_store, claims_repo):
    return {t.name: t for t in build_tools(policy_store, claims_repo)}


def test_get_claim_status_found(tools):
    out = json.loads(tools["get_claim_status"].invoke({"claim_id": "clm-8821"}))
    assert out["ok"] is True
    assert out["claim"]["status"] == "Approved"


def test_get_claim_status_not_found(tools):
    out = json.loads(tools["get_claim_status"].invoke({"claim_id": "CLM-0000"}))
    assert out == {"ok": False, "error": "not_found", "claim_id": "CLM-0000"}


def test_get_claim_status_rejects_malformed_id(tools):
    out = json.loads(tools["get_claim_status"].invoke({"claim_id": "8821; DROP TABLE"}))
    assert out["error"] == "validation_error"


def test_submit_claim_appends_record(tools, claims_path):
    out = json.loads(
        tools["submit_claim"].invoke(
            {
                "policy_number": "POL-1092",
                "claim_type": "water damage",
                "amount": 1800.5,
                "description": "Pipe burst under the kitchen sink.",
            }
        )
    )
    assert out["ok"] is True
    records = read_claims(claims_path)
    assert len(records) == 3
    new = records[-1]
    assert new["claim_id"] == out["confirmation_id"]
    assert new["claim_type"] == "Water Damage"
    assert new["status"] == "Submitted"
    assert new["amount"] == 1800.5


@pytest.mark.parametrize(
    "overrides, bad_field",
    [
        ({"policy_number": "1092"}, "policy_number"),
        ({"claim_type": "Alien Abduction"}, "claim_type"),
        ({"amount": -5}, "amount"),
        ({"amount": 12.345}, "amount"),
        ({"description": "short"}, "description"),
    ],
)
def test_submit_claim_validation(tools, claims_path, overrides, bad_field):
    args = {
        "policy_number": "POL-1092",
        "claim_type": "Personal Property",
        "amount": 500,
        "description": "Laptop stolen from my car.",
    } | overrides
    out = json.loads(tools["submit_claim"].invoke(args))
    assert out["ok"] is False
    assert any(d.startswith(bad_field) for d in out["details"])
    assert len(read_claims(claims_path)) == 2  # nothing written


def test_submission_model_forbids_extra_fields():
    with pytest.raises(ValidationError):
        ClaimSubmission(
            policy_number="POL-1092",
            claim_type="Water Damage",
            amount=10,
            description="A valid description",
            status="Approved",
        )


def test_repository_round_trip(claims_repo):
    claim = claims_repo.submit(
        ClaimSubmission(policy_number="pol-3341", claim_type="Personal Property", amount=99, description="Broken TV screen")
    )
    assert claims_repo.get(claim.claim_id).policy_number == "POL-3341"
    with pytest.raises(ClaimNotFoundError):
        claims_repo.get("CLM-0001")


def test_search_policy_returns_citations(tools):
    msg = tools["search_policy"].invoke(
        {"type": "tool_call", "id": "1", "name": "search_policy", "args": {"query": "burst pipe water damage"}}
    )
    assert "[1] (sample_policy.md — Section 1: Home Water Damage Coverage)" in msg.content
    assert msg.artifact[0]["section"] == "Section 1: Home Water Damage Coverage"
