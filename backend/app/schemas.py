"""Public API request/response models."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated

UserId = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9_\-]{1,64}$")]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {"user_id": "usr_123", "message": "Is a burst pipe covered?"}})

    user_id: UserId
    # Upper bound is enforced against settings in the route; this is a hard ceiling.
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    output: Any = None
    status: str = "success"


class ChatResponse(BaseModel):
    response: str
    sources: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"
