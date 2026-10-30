"""Domain enums for audio-prompt-engine."""

from enum import Enum


class OutputMode(str, Enum):
    """Output mode requested by the client."""

    TXT = "txt"
    PROMPT = "prompt"


class PromptTechnique(str, Enum):
    """Supported prompt engineering techniques."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ROLE_PROMPTING = "role_prompting"
    CO_STAR = "co_star"
    REACT = "react"


class WhisperModelSize(str, Enum):
    """Supported Whisper model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V3 = "large-v3"


class UserRole(str, Enum):
    """Role-based access control roles."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


class LLMProvider(str, Enum):
    """Supported LLM inference providers."""

    OLLAMA = "ollama"
    GROK = "grok"

