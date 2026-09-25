import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.graph import ClaimsAgent
from app.agent.llm import build_llm
from app.agent.tools import build_tools
from app.api.routes import router
from app.config import Settings, get_settings
from app.rag.embeddings import build_embeddings
from app.rag.store import PolicyStore
from app.tools.claims import ClaimsRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_agent(settings: Settings) -> ClaimsAgent:
    store = PolicyStore(build_embeddings(settings), settings.chroma_persist_dir)
    count = store.ingest(settings.policy_docs_dir)
    logger.info("Indexed %d policy chunks from %s", count, settings.policy_docs_dir)
    tools = build_tools(
        store,
        ClaimsRepository(settings.claims_db_path),
        retrieval_k=settings.retrieval_k,
        min_score=settings.retrieval_min_score,
        max_gap=settings.retrieval_max_gap,
    )
    return ClaimsAgent(build_llm(settings), tools)


def create_app(agent: ClaimsAgent | None = None) -> FastAPI:
    """Pass `agent` to inject a pre-built agent (used by tests)."""
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.agent = agent or build_agent(settings)
        yield

    app = FastAPI(title="OmniCare Claims Assistant", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
