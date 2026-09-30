Written for: developers new to the claim-rag codebase.

The graph matches the current commit (`9ef231f`). The only working-tree change is your `.gitignore` edit adding `.ua/`, which doesn't affect the code, so the guide below is up to date.

---

# claim-rag Onboarding Guide

## Project Overview

**claim-rag** is a prototype customer assistant for OmniCare Financial, an insurance company. It does three things:

1. **Answers policy coverage questions**, citing the policy sections it used. Answers come from RAG, meaning the relevant sections are retrieved from `sample_policy.md` and given to the model.
2. **Looks up a claim's status** from a JSON "database".
3. **Submits new claims**, after validating every field.

| | |
|---|---|
| **Languages** | Python (all code), plus Markdown, JSON, YAML and Dockerfiles |
| **Backend** | FastAPI, Pydantic, LangGraph and LangChain, Chroma (a local vector store) and fastembed (local embeddings) |
| **Frontend** | Streamlit, calling the backend with httpx |
| **LLM** | Any tool-calling model: Groq (the default), OpenAI, Anthropic or local Ollama |
| **Tests** | pytest, fully offline, using a scripted fake LLM |
| **Run** | Docker Compose: API on `:8000`, UI on `:8501` |

**How one message flows through the system:**

```
Streamlit UI ──POST /api/v1/chat──▶ FastAPI route ──▶ ClaimsAgent.chat(user_id, msg)
                                                          │
                                   LangGraph:  guard ──blocked──▶ refusal (LLM never called)
                                                 │ ok
                                               agent (LLM) ⇄ tools
                                                          ├─ search_policy    → Chroma → sample_policy.md
                                                          ├─ get_claim_status → mock_claims.json
                                                          └─ submit_claim     → mock_claims.json
```

## Architecture Layers

| Layer | What it does | Key files |
|---|---|---|
| **Streamlit Frontend** | The chat UI and its HTTP client | `frontend/app.py`, `frontend/api_client.py` |
| **API Layer** | The FastAPI app factory, routes and request/response schemas | `backend/app/main.py`, `api/routes.py`, `schemas.py` |
| **Agent Orchestration** | The LangGraph workflow, LLM setup, system prompt, tools and guardrails | `agent/graph.py`, `agent/tools.py`, `agent/llm.py`, `agent/prompts.py`, `safety/guardrails.py` |
| **Retrieval & Claims Data** | Policy search over Chroma, the embedding models and the JSON claims store | `rag/store.py`, `rag/embeddings.py`, `tools/claims.py`, `data/*` |
| **Configuration & Docs** | Settings, the env template, dependency lists, pytest config and the README | `app/config.py`, `.env.example`, `requirements*.txt`, `pytest.ini`, `README.md` |
| **Test Layer** | Offline test suites for the backend and frontend | `backend/tests/*`, `frontend/tests/test_api_client.py` |
| **Infrastructure** | Container images and the compose stack | `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` |

## Key Concepts

- **The guardrail is a real step in the graph.** `guard` runs before the LLM. It normalizes the text, strips invisible characters, and blocks prompt-injection patterns. A blocked message ends the turn with a fixed refusal, and the LLM is never called.
- **Tool results are recorded in the graph state.** Every tool call and its result is saved in the conversation. After each turn, `_to_result` in `graph.py` reads them back to build the API's `sources` and `tool_calls`. If you change a tool's output format, check that function.
- **Conversation memory is per user and in memory only.** LangGraph's `InMemorySaver` keeps one conversation per `user_id`. That's what lets a claim be filed over several messages. It's lost when the backend restarts, and it isn't shared between processes. The UI's **New conversation** button switches to a fresh `user_id`.
- **Validation errors go back to the LLM.** The tools check their arguments with strict Pydantic models (ID formats, the claim type, amount limits, no extra fields). A failure comes back as a JSON error, so the model can ask the user to correct it rather than crashing.
- **Search results are filtered by relevance.** `PolicyStore.search` drops results below a minimum score, `RETRIEVAL_MIN_SCORE`. It also drops results scoring more than `RETRIEVAL_MAX_GAP` below the best match. Both values were tuned for the bge-small embedding model on the sample policy, so re-measure them if you change the model or the documents.
- **Every setting lives in one place.** `config.py` reads them all (the LLM provider and model, the embedding backend, thresholds, file paths) from environment variables or `.env`.
- **Tests run offline.** A scripted fake chat model drives the real LangGraph graph, and a simple "hashing" embedder replaces the real embedding model. Tests need no API key and no model download.
- **Claims are written safely.** The claims JSON is written to a temp file and then renamed over the original, under a lock. In Docker it lives on a named volume, seeded from `mock_claims.json` the first time.

## Guided Tour

1. **Project overview:** `README.md`. It covers the data flow, why LangGraph was chosen, and the safety layers.
2. **The chat UI:** `frontend/app.py` handles session state and rendering. `api_client.py` wraps the HTTP calls and is kept separate from the UI so it can be tested.
3. **Backend entry and settings:** `main.py` wires the store, embeddings, repository, tools and LLM into a `ClaimsAgent`. `config.py` and `.env.example` hold the settings.
4. **API contract:** `routes.py` defines `POST /api/v1/chat` and `GET /api/v1/health`. `schemas.py` defines the request and response shapes.
5. **The LangGraph workflow:** `graph.py` defines the `guard → agent ⇄ tools` graph. `llm.py` creates the chat model for the chosen provider, and `prompts.py` holds the system prompt.
6. **Input guardrails:** `guardrails.py` normalizes text and checks it against the injection patterns.
7. **Agent tools:** `tools.py` defines `search_policy`, `get_claim_status` and `submit_claim`.
8. **Policy search (RAG):** `store.py` splits the policy into one chunk per section and filters results by relevance. `embeddings.py` provides the fastembed model and the offline hashing embedder, and `sample_policy.md` is the policy being searched.
9. **Claims data:** `claims.py` holds the Pydantic models and the claims repository. `mock_claims.json` is the seed data.
10. **Tests:** `conftest.py` provides the fixtures. `test_api.py`, `test_guardrails.py`, `test_rag.py` and `test_tools.py` hold the suites.
11. **Containers:** `docker-compose.yml` and the two Dockerfiles. The backend image downloads the embedding model at build time.

## File Map

**Frontend**
- `frontend/app.py`: the Streamlit chat page. It holds the user ID and message history, and shows tool-call and source expanders, a backend health badge and example prompts.
- `frontend/api_client.py`: `BackendClient`, which wraps `/health` and `/chat` and turns HTTP and network failures into readable `BackendError` messages.

**API**
- `backend/app/main.py`: the app factory (`create_app`) and startup wiring (`build_agent`). Tests can pass in a ready-made agent.
- `backend/app/api/routes.py`: the health and chat endpoints. The chat endpoint enforces the message length limit and returns 502 if the agent fails.
- `backend/app/schemas.py`: `ChatRequest`, `ChatResponse`, `ToolCall` and `HealthResponse`.

**Agent**
- `backend/app/agent/graph.py`: `build_graph`, the `ClaimsAgent` class the API calls, and the code that extracts each turn's result.
- `backend/app/agent/tools.py`: `build_tools`, which defines the three tools and their argument schemas.
- `backend/app/agent/llm.py`: `build_llm`, which creates the chat model for the configured provider and only imports that provider's package.
- `backend/app/agent/prompts.py`: the system prompt, covering the assistant's scope, citation rules and how to treat tool text.
- `backend/app/safety/guardrails.py`: `normalize`, `check_user_input` and the refusal message.

**Retrieval and data**
- `backend/app/rag/store.py`: `load_policy_chunks` and `PolicyStore` (loading the policy into the index and searching it).
- `backend/app/rag/embeddings.py`: `FastEmbedEmbeddings` (real runs) and `HashingEmbeddings` (tests and offline use).
- `backend/app/tools/claims.py`: the claim models and the thread-safe `ClaimsRepository`.
- `backend/data/sample_policy.md` and `backend/data/mock_claims.json`: the searched policy and the seed claims.

**Configuration**
- `backend/app/config.py`: all settings. `backend/.env.example` is the template you copy to `.env`.

## Complexity Hotspots

Nothing in the codebase is rated `complex`. These are the `moderate` areas worth reading carefully before you change them:

| Where | Why to be careful |
|---|---|
| `agent/graph.py` → `build_graph`, `_to_result` | The routing (guard → end, the tool loop) and how each turn's `sources` and `tool_calls` are extracted. A change to a tool's output breaks the API response here. |
| `agent/tools.py` → `build_tools` | This is where the LLM's loosely typed arguments meet the strict Pydantic models. Validation errors must stay JSON the model can read. |
| `rag/store.py` → `PolicyStore` | The chunk IDs (which prevent duplicate entries on re-indexing) and the two relevance rules. The thresholds depend on the embedding model. |
| `tools/claims.py` → `ClaimsRepository` | Locking, atomic writes, and ID patterns that accept lowercase input. |
| `main.py` | Startup wiring. Every setting has to be passed through here into the tools and the agent. |
| `backend/tests/conftest.py` | The fake LLM plays back scripted replies in order, so each test's script must match exactly the model calls the graph will make. |
| `frontend/app.py` | Streamlit reruns the whole script on every interaction. Anything that changes a widget's value (like resetting `user_id`) must happen in an `on_click` callback. |

**Getting started:** copy `backend/.env.example` to `backend/.env`, add a free Groq key, run `docker compose up --build`, then run `pytest` in `backend/` and `frontend/`.

---

Do you want me to save this as `docs/UA_ONBOARDING.md`? If you commit it (along with the pending `.gitignore` change), the rest of the team can use it without the graph tooling.