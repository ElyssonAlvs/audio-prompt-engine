"""Configuration management using pydantic-settings."""

import json
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.domain.enums import LLMProvider, UserRole, WhisperModelSize


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str = Field(default="development", description="Application environment")
    DEBUG: bool = Field(default=False, description="Debug flag")
    HOST: str = Field(default="127.0.0.1", description="HTTP host")
    PORT: int = Field(default=8000, description="HTTP port")

    DATABASE_PATH: str = Field(default="audio_prompt_engine.db", description="SQLite database path")

    WHISPER_MODEL_SIZE: WhisperModelSize = Field(
        default=WhisperModelSize.BASE, description="Whisper model size"
    )
    WHISPER_DEVICE: str = Field(default="cpu", description="Inference device: cpu or cuda")
    WHISPER_COMPUTE_TYPE: str = Field(default="int8", description="Quantization compute type")

    MAX_UPLOAD_SIZE_BYTES: int = Field(
        default=25 * 1024 * 1024, description="Maximum audio upload size in bytes (25MB)"
    )

    # LLM Execution Engine Configuration (Ollama & Grok)
    OLLAMA_BASE_URL: str = Field(
        default="http://host.docker.internal:11434", description="Ollama API base URL"
    )
    XAI_API_KEY: Optional[str] = Field(
        default=None, description="xAI API Key for Grok inference"
    )
    DEFAULT_GROK_MODEL: str = Field(
        default="grok-beta", description="Default model for Grok inference"
    )
    DEFAULT_OLLAMA_MODEL: str = Field(
        default="llama3", description="Default model for Ollama inference"
    )
    DEFAULT_LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.OLLAMA, description="Default LLM provider"
    )

    # API Keys mapping: api_key -> {tenant_id: str, role: UserRole}
    API_KEYS_CONFIG: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "dev-operator-alpha": {"tenant_id": "tenant-alpha", "role": UserRole.OPERATOR.value},
            "dev-viewer-alpha": {"tenant_id": "tenant-alpha", "role": UserRole.VIEWER.value},
            "dev-operator-beta": {"tenant_id": "tenant-beta", "role": UserRole.OPERATOR.value},
            "dev-admin": {"tenant_id": "tenant-admin", "role": UserRole.ADMIN.value},
        }
    )

    @field_validator("API_KEYS_CONFIG", mode="before")
    @classmethod
    def parse_api_keys(cls, v: Any) -> Dict[str, Dict[str, Any]]:
        """Parse JSON string into dictionary if loaded from .env."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        if isinstance(v, dict):
            return v
        return {
            "dev-operator-alpha": {"tenant_id": "tenant-alpha", "role": UserRole.OPERATOR.value},
            "dev-viewer-alpha": {"tenant_id": "tenant-alpha", "role": UserRole.VIEWER.value},
            "dev-operator-beta": {"tenant_id": "tenant-beta", "role": UserRole.OPERATOR.value},
            "dev-admin": {"tenant_id": "tenant-admin", "role": UserRole.ADMIN.value},
        }

    def authenticate_key(self, api_key: Optional[str]) -> Optional[Tuple[str, UserRole]]:
        """Authenticate an API Key and return (tenant_id, role) or None."""
        if not api_key:
            return None
        info = self.API_KEYS_CONFIG.get(api_key)
        if not info:
            return None
        tenant_id = info.get("tenant_id")
        role_str = info.get("role", "viewer")
        try:
            role = UserRole(role_str)
        except ValueError:
            role = UserRole.VIEWER
        if not tenant_id:
            return None
        return str(tenant_id), role


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
