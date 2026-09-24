import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.agent.graph import ClaimsAgent
from app.config import Settings, get_settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse, ToolCall

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


def get_agent(request: Request) -> ClaimsAgent:
    return request.app.state.agent


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    agent: ClaimsAgent = Depends(get_agent),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    if len(body.message) > settings.max_message_chars:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"message exceeds {settings.max_message_chars} characters",
        )
    try:
        result = await agent.chat(body.user_id, body.message)
    except Exception:
        logger.exception("Agent failed for user %s", body.user_id)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "The assistant is temporarily unavailable.") from None

    if result.blocked:
        logger.warning("Guardrail blocked a message from user %s", body.user_id)
    return ChatResponse(
        response=result.response,
        sources=result.sources,
        tool_calls=[ToolCall(**tc) for tc in result.tool_calls],
    )
