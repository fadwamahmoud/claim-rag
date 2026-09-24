"""Chat model factory. Provider packages are imported lazily so only the one in use must be configured."""

from langchain_core.language_models import BaseChatModel

from app.config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    common = {"model": settings.llm_model, "temperature": settings.llm_temperature}
    match settings.llm_provider:
        case "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(**common)  # reads GROQ_API_KEY
        case "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(**common)  # reads OPENAI_API_KEY
        case "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(**common)  # reads ANTHROPIC_API_KEY
        case "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(**common, base_url=settings.ollama_base_url)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
