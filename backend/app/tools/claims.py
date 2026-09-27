"""Mock claims backend: a JSON file acting as the claims database.

The functions here are plain Python so they can be unit-tested directly; the
LangChain tool wrappers the agent uses live in `app.agent.tools`.
"""

import json
import os
import secrets
import tempfile
import threading
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

ClaimId = Annotated[
    str, StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"(?i)^CLM-\d{4,8}$")
]
PolicyNumber = Annotated[
    str, StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"(?i)^POL-\d{4,8}$")
]


class ClaimType(str, Enum):
    WATER_DAMAGE = "Water Damage"
    PERSONAL_PROPERTY = "Personal Property"


class ClaimStatus(str, Enum):
    SUBMITTED = "Submitted"
    UNDER_REVIEW = "Under Review"
    APPROVED = "Approved"
    DENIED = "Denied"


class ClaimLookup(BaseModel):
    claim_id: ClaimId


class ClaimSubmission(BaseModel):
    """Input for `submit_claim`. Strict: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    policy_number: PolicyNumber
    claim_type: ClaimType
    amount: Decimal = Field(gt=0, le=1_000_000, max_digits=12, decimal_places=2)
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=1000)]

    @field_validator("claim_type", mode="before")
    @classmethod
    def _normalise_claim_type(cls, value: object) -> object:
        # Accept "water damage", "WATER_DAMAGE", etc.
        if isinstance(value, str):
            key = value.strip().replace("_", " ").casefold()
            for member in ClaimType:
                if member.value.casefold() == key:
                    return member
        return value


class Claim(BaseModel):
    claim_id: str
    policy_number: str
    claim_type: str
    status: str
    amount: float
    description: str | None = None


class ClaimNotFoundError(LookupError):
    pass


class ClaimsRepository:
    """Thread-safe read/append access to the JSON claims file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, records: list[dict]) -> None:
        # Write to a temp file and rename so a crash never leaves a half-written DB.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2)
                fh.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get(self, claim_id: str) -> Claim:
        lookup = ClaimLookup(claim_id=claim_id)
        with self._lock:
            records = self._read()
        for record in records:
            if record.get("claim_id", "").upper() == lookup.claim_id:
                return Claim.model_validate(record)
        raise ClaimNotFoundError(lookup.claim_id)

    def submit(self, submission: ClaimSubmission) -> Claim:
        with self._lock:
            records = self._read()
            existing = {r.get("claim_id") for r in records}
            claim_id = _new_claim_id(existing)
            claim = Claim(
                claim_id=claim_id,
                policy_number=submission.policy_number,
                claim_type=submission.claim_type.value,
                status=ClaimStatus.SUBMITTED.value,
                amount=float(submission.amount),
                description=submission.description,
            )
            records.append(claim.model_dump(exclude_none=True))
            self._write(records)
        return claim


def _new_claim_id(existing: set[str]) -> str:
    while True:
        candidate = f"CLM-{secrets.randbelow(90000) + 10000}"
        if candidate not in existing:
            return candidate
