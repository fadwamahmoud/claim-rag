"""LangChain tools exposed to the agent.

Tool arguments are declared loosely here and validated inside each tool with the
strict Pydantic models from `app.tools.claims`, so a bad value from the LLM comes
back as a readable error the model can relay or correct, not an exception.
"""

import json

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, ValidationError

from app.rag.store import PolicyStore
from app.tools.claims import ClaimNotFoundError, ClaimsRepository, ClaimSubmission


class SearchPolicyArgs(BaseModel):
    query: str = Field(description="A focused search query about policy coverage, limits, deductibles or exclusions.")


class ClaimStatusArgs(BaseModel):
    claim_id: str = Field(description="Claim identifier, e.g. CLM-8821.")


class SubmitClaimArgs(BaseModel):
    policy_number: str = Field(description="Policy number, e.g. POL-1092.")
    claim_type: str = Field(description="Either 'Water Damage' or 'Personal Property'.")
    amount: float = Field(description="Claimed amount in USD, greater than 0.")
    description: str = Field(description="What happened, in the policyholder's words (10-1000 characters).")


def _validation_error(exc: ValidationError) -> str:
    problems = [f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors()]
    return json.dumps({"ok": False, "error": "validation_error", "details": problems})


def build_tools(
    policy_store: PolicyStore,
    claims: ClaimsRepository,
    retrieval_k: int = 3,
    min_score: float = 0.0,
    max_gap: float = 1.0,
) -> list[BaseTool]:
    def search_policy(query: str) -> tuple[str, list[dict]]:
        chunks = policy_store.search(query, k=retrieval_k, min_score=min_score, max_gap=max_gap)
        if not chunks:
            return "No relevant policy sections found.", []
        blocks = [f"[{i}] ({c.citation})\n{c.content}" for i, c in enumerate(chunks, start=1)]
        artifact = [
            {"citation": c.citation, "source": c.source, "section": c.section, "score": c.score}
            for c in chunks
        ]
        return "\n\n".join(blocks), artifact

    def get_claim_status(claim_id: str) -> str:
        try:
            claim = claims.get(claim_id)
        except ValidationError as exc:
            return _validation_error(exc)
        except ClaimNotFoundError:
            return json.dumps({"ok": False, "error": "not_found", "claim_id": claim_id.strip().upper()})
        return json.dumps({"ok": True, "claim": claim.model_dump(exclude_none=True)})

    def submit_claim(policy_number: str, claim_type: str, amount: float, description: str) -> str:
        try:
            submission = ClaimSubmission(
                policy_number=policy_number, claim_type=claim_type, amount=amount, description=description
            )
        except ValidationError as exc:
            return _validation_error(exc)
        claim = claims.submit(submission)
        return json.dumps({"ok": True, "confirmation_id": claim.claim_id, "claim": claim.model_dump()})

    return [
        StructuredTool.from_function(
            func=search_policy,
            name="search_policy",
            description="Search the OmniCare policy documents. Use for ANY question about what is or isn't covered.",
            args_schema=SearchPolicyArgs,
            response_format="content_and_artifact",
        ),
        StructuredTool.from_function(
            func=get_claim_status,
            name="get_claim_status",
            description="Look up the current status of an existing insurance claim by its claim ID.",
            args_schema=ClaimStatusArgs,
        ),
        StructuredTool.from_function(
            func=submit_claim,
            name="submit_claim",
            description=(
                "Submit a NEW insurance claim. Only call once the user has explicitly provided all four fields; "
                "never guess or invent values."
            ),
            args_schema=SubmitClaimArgs,
        ),
    ]
