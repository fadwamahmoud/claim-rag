"""Input guardrails: cheap, deterministic checks that run before the LLM.

This is one layer of defence. The system prompt also tells the model to treat
retrieved text and tool output as data, and every tool argument is validated
with Pydantic, so an injection that slips past the heuristics still cannot make
a tool accept malformed input.
"""

import re
import unicodedata
from dataclasses import dataclass

_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your|system)\b.{0,40}\b(instructions?|prompts?|rules?|guidelines?|directives?)",
    r"\b(reveal|show|print|repeat|leak|output)\b.{0,40}\b(system|hidden|initial|original)\s+(prompt|instructions?|message)",
    r"\byou\s+are\s+now\b",
    r"\b(act|behave|pretend|roleplay)\s+(as|like)\b.{0,40}\b(unrestricted|jailbroken|dan|developer\s+mode|admin|system)",
    r"\b(developer|god|jailbreak|dan)\s+mode\b",
    r"\bnew\s+(system\s+)?instructions?\s*:",
    r"(^|\n)\s*(system|assistant)\s*:",
    r"<\|?\s*(im_start|im_end|system|endoftext)\s*\|?>",
    r"\[/?(INST|SYS)\]",
    r"\b(set|change|update|mark)\b.{0,30}\b(claim|status)\b.{0,30}\b(to|as)\s+approved\b",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_CONTROL_CHARS = {"Cc", "Cf"}  # control + format chars (zero-width, bidi overrides)


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str | None = None


def normalize(text: str) -> str:
    """NFKC-fold and strip invisible characters that are used to hide payloads."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) not in _CONTROL_CHARS)


def check_user_input(text: str) -> GuardrailResult:
    cleaned = normalize(text)
    if not cleaned.strip():
        return GuardrailResult(False, "empty_message")
    for pattern in _INJECTION_RE:
        if pattern.search(cleaned):
            return GuardrailResult(False, "prompt_injection")
    return GuardrailResult(True)


REFUSAL_MESSAGE = (
    "I can't help with that request. I can answer questions about your OmniCare "
    "policy coverage, check the status of a claim, or help you submit a new claim."
)
