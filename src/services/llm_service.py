"""LLM Execution Engine service supporting Ollama (local) and Grok (xAI cloud)."""

import json
import logging
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from fastapi import HTTPException, status

from src.config import Settings, get_settings
from src.domain.enums import LLMProvider
from src.domain.schemas import LLMExecutionRequest, LLMExecutionResponse

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60.0
OLLAMA_TIMEOUT_SECONDS = 180.0


class ILLMProvider(ABC):
    """Abstract Strategy interface for LLM inference providers."""

    @abstractmethod
    def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute inference against the LLM provider.

        Returns:
            Dict with keys:
            - response_text: str
            - model: str
            - tokens_used: Optional[int]
            - execution_time_seconds: float
        """
        pass


class OllamaProvider(ILLMProvider):
    """Provider for local Ollama instances (offline, private, 0 cost)."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.OLLAMA_BASE_URL.rstrip("/")
        self.default_model = self.settings.DEFAULT_OLLAMA_MODEL

    def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        chosen_model = model or self.default_model
        endpoint = f"{self.base_url}/api/generate"

        payload = {
            "model": chosen_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else 0.7,
                "num_predict": 512,
            },
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
        except (TimeoutError, socket.timeout):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Timeout comunicando com Ollama em {self.base_url} (limite de {int(OLLAMA_TIMEOUT_SECONDS)}s excedido).",
            )
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            logger.error(f"Ollama returned HTTP error {e.code}: {error_body}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ollama retornou erro HTTP {e.code}: {e.reason}.",
            )
        except urllib.error.URLError as e:
            if isinstance(e.reason, (TimeoutError, socket.timeout)) or "timed out" in str(e.reason).lower():
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=f"Timeout comunicando com Ollama em {self.base_url}.",
                )
            logger.error(f"Cannot connect to Ollama at {self.base_url}: {e.reason}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Não foi possível conectar ao Ollama em {self.base_url}. Verifique se o serviço está em execução.",
            )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Resposta inválida (não-JSON) recebida do Ollama.",
            )

        elapsed = round(time.perf_counter() - start_time, 3)

        response_text = data.get("response", "")
        tokens = None
        eval_count = data.get("eval_count")
        prompt_eval = data.get("prompt_eval_count")
        if eval_count is not None or prompt_eval is not None:
            tokens = (eval_count or 0) + (prompt_eval or 0)

        return {
            "response_text": response_text,
            "model": chosen_model,
            "tokens_used": tokens,
            "execution_time_seconds": elapsed,
        }


class GrokProvider(ILLMProvider):
    """Provider for xAI Grok Cloud API (70B deep analytical capability)."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_key = self.settings.XAI_API_KEY
        self.default_model = self.settings.DEFAULT_GROK_MODEL
        self.endpoint = "https://api.x.ai/v1/chat/completions"

    def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        active_key = api_key or self.api_key
        if not active_key:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="XAI_API_KEY não configurada no servidor. Configure a chave para executar inferência via Grok.",
            )

        chosen_model = model or self.default_model

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "model": chosen_model,
            "stream": False,
            "temperature": temperature if temperature is not None else 0.7,
        }

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {active_key}",
                "User-Agent": "audio-prompt-engine/0.1.0",
            },
            method="POST",
        )

        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
        except (TimeoutError, socket.timeout):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Timeout comunicando com a API do Grok (limite de {int(DEFAULT_TIMEOUT_SECONDS)}s excedido).",
            )
        except urllib.error.HTTPError as e:
            if e.code == 401:
                logger.error("xAI Grok authentication failed: invalid API key.")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Falha de autenticação com a API do Grok (chave inválida ou revogada).",
                )
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            logger.error(f"Grok returned HTTP error {e.code}: {error_body}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Grok retornou erro HTTP {e.code}: {e.reason}.",
            )
        except urllib.error.URLError as e:
            if isinstance(e.reason, (TimeoutError, socket.timeout)) or "timed out" in str(e.reason).lower():
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Timeout comunicando com a API do Grok.",
                )
            logger.error(f"Cannot connect to Grok API: {e.reason}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Não foi possível conectar à API do Grok: {e.reason}.",
            )
        except (json.JSONDecodeError, KeyError, IndexError):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Formato de resposta inesperado recebido da API do Grok.",
            )

        elapsed = round(time.perf_counter() - start_time, 3)

        try:
            response_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            response_text = ""

        tokens = data.get("usage", {}).get("total_tokens")

        return {
            "response_text": response_text,
            "model": chosen_model,
            "tokens_used": tokens,
            "execution_time_seconds": elapsed,
        }


class GeminiProvider(ILLMProvider):
    """Provider for Google AI Studio Gemini API (gemini-1.5-flash)."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_key = self.settings.GEMINI_API_KEY
        self.default_model = self.settings.DEFAULT_GEMINI_MODEL
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        active_key = api_key or self.api_key
        if not active_key:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="GEMINI_API_KEY não configurada no servidor. Configure a chave para executar inferência via Gemini.",
            )

        chosen_model = model or self.default_model
        endpoint = f"{self.base_url}/{chosen_model}:generateContent?key={active_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature if temperature is not None else 0.7,
            },
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
        except (TimeoutError, socket.timeout):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Timeout comunicando com a API do Gemini (limite de {int(DEFAULT_TIMEOUT_SECONDS)}s excedido).",
            )
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403):
                logger.error(f"Gemini API authentication/access error (HTTP {e.code}).")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Falha de autenticação ou acesso com a API do Gemini (HTTP {e.code}).",
                )
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            logger.error(f"Gemini returned HTTP error {e.code}: {error_body}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini retornou erro HTTP {e.code}: {e.reason}.",
            )
        except urllib.error.URLError as e:
            if isinstance(e.reason, (TimeoutError, socket.timeout)) or "timed out" in str(e.reason).lower():
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Timeout comunicando com a API do Gemini.",
                )
            logger.error(f"Cannot connect to Gemini API: {e.reason}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Não foi possível conectar à API do Gemini: {e.reason}.",
            )
        except (json.JSONDecodeError, KeyError, IndexError):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Formato de resposta inesperado recebido da API do Gemini.",
            )

        elapsed = round(time.perf_counter() - start_time, 3)

        try:
            response_text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            response_text = ""

        tokens = data.get("usageMetadata", {}).get("totalTokenCount")

        return {
            "response_text": response_text,
            "model": chosen_model,
            "tokens_used": tokens,
            "execution_time_seconds": elapsed,
        }


class GroqProvider(ILLMProvider):
    """Provider for Groq Cloud LPU API (llama-3.3-70b-versatile ultra-fast inference)."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_key = self.settings.GROQ_API_KEY
        self.default_model = self.settings.DEFAULT_GROQ_MODEL
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        active_key = api_key or self.api_key
        if not active_key:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="GROQ_API_KEY não configurada no servidor. Configure a chave para executar inferência via Groq.",
            )

        chosen_model = model or self.default_model

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "model": chosen_model,
            "stream": False,
            "temperature": temperature if temperature is not None else 0.7,
        }

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {active_key}",
                "User-Agent": "audio-prompt-engine/0.1.0",
            },
            method="POST",
        )

        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
        except (TimeoutError, socket.timeout):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Timeout comunicando com a API do Groq (limite de {int(DEFAULT_TIMEOUT_SECONDS)}s excedido).",
            )
        except urllib.error.HTTPError as e:
            if e.code == 401:
                logger.error("Groq authentication failed: invalid API key.")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Falha de autenticação com a API do Groq (chave inválida ou revogada).",
                )
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            logger.error(f"Groq returned HTTP error {e.code}: {error_body}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Groq retornou erro HTTP {e.code}: {e.reason}.",
            )
        except urllib.error.URLError as e:
            if isinstance(e.reason, (TimeoutError, socket.timeout)) or "timed out" in str(e.reason).lower():
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Timeout comunicando com a API do Groq.",
                )
            logger.error(f"Cannot connect to Groq API: {e.reason}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Não foi possível conectar à API do Groq: {e.reason}.",
            )
        except (json.JSONDecodeError, KeyError, IndexError):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Formato de resposta inesperado recebido da API do Groq.",
            )

        elapsed = round(time.perf_counter() - start_time, 3)

        try:
            response_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            response_text = ""

        tokens = data.get("usage", {}).get("total_tokens")

        return {
            "response_text": response_text,
            "model": chosen_model,
            "tokens_used": tokens,
            "execution_time_seconds": elapsed,
        }


class LLMExecutionService:
    """Orchestrates prompt execution across supported LLM providers."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        ollama_provider: Optional[ILLMProvider] = None,
        grok_provider: Optional[ILLMProvider] = None,
        gemini_provider: Optional[ILLMProvider] = None,
        groq_provider: Optional[ILLMProvider] = None,
    ):
        self.settings = settings or get_settings()
        self._providers: Dict[LLMProvider, ILLMProvider] = {
            LLMProvider.OLLAMA: ollama_provider or OllamaProvider(settings=self.settings),
            LLMProvider.GROK: grok_provider or GrokProvider(settings=self.settings),
            LLMProvider.GEMINI: gemini_provider or GeminiProvider(settings=self.settings),
            LLMProvider.GROQ: groq_provider or GroqProvider(settings=self.settings),
        }

    def set_provider(self, provider_type: LLMProvider, provider: ILLMProvider) -> None:
        """Inject or override a provider implementation (useful for testing)."""
        self._providers[provider_type] = provider

    def execute(
        self,
        job_id: str,
        prompt: str,
        request: LLMExecutionRequest,
    ) -> LLMExecutionResponse:
        """Execute a prompt against the selected LLM provider."""
        provider = self._providers.get(request.provider)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provedor LLM '{request.provider}' não é suportado.",
            )

        prompt_to_run = request.prompt_override if request.prompt_override else prompt

        result = provider.execute(
            prompt=prompt_to_run,
            model=request.model,
            temperature=request.temperature,
            api_key=request.api_key,
        )

        return LLMExecutionResponse(
            job_id=job_id,
            provider=request.provider,
            model=result["model"],
            prompt_executed=prompt_to_run,
            response_text=result["response_text"],
            execution_time_seconds=result["execution_time_seconds"],
            tokens_used=result.get("tokens_used"),
        )


_llm_service_instance: Optional[LLMExecutionService] = None


def get_llm_service() -> LLMExecutionService:
    """Return singleton LLMExecutionService instance."""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMExecutionService()
    return _llm_service_instance
