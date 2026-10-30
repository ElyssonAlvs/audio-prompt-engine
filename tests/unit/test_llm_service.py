"""Unit tests for the LLM Execution Service, OllamaProvider, and GrokProvider."""

import io
import json
import socket
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.config import Settings
from src.domain.enums import LLMProvider
from src.domain.schemas import LLMExecutionRequest
from src.services.llm_service import (
    GrokProvider,
    LLMExecutionService,
    OllamaProvider,
)


class TestOllamaProvider:
    """Unit tests for the local Ollama provider."""

    def test_ollama_successful_execution(self):
        settings = Settings(
            OLLAMA_BASE_URL="http://localhost:11434",
            DEFAULT_OLLAMA_MODEL="llama3",
        )
        provider = OllamaProvider(settings=settings)

        mock_response_data = {
            "model": "llama3",
            "response": "Refatoração concluída com sucesso.",
            "eval_count": 45,
            "prompt_eval_count": 15,
        }
        mock_response_bytes = json.dumps(mock_response_data).encode("utf-8")

        mock_cm = MagicMock()
        mock_cm.read.return_value = mock_response_bytes
        mock_cm.__enter__.return_value = mock_cm

        with patch("urllib.request.urlopen", return_value=mock_cm) as mock_urlopen:
            result = provider.execute(prompt="Refatore esta função", model="llama3", temperature=0.5)

            assert result["response_text"] == "Refatoração concluída com sucesso."
            assert result["model"] == "llama3"
            assert result["tokens_used"] == 60
            assert result["execution_time_seconds"] >= 0.0

            # Verify request payload
            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "http://localhost:11434/api/generate"
            sent_body = json.loads(req.data.decode("utf-8"))
            assert sent_body["model"] == "llama3"
            assert sent_body["prompt"] == "Refatore esta função"
            assert sent_body["options"]["temperature"] == 0.5

    def test_ollama_timeout_raises_504(self):
        provider = OllamaProvider()
        with patch("urllib.request.urlopen", side_effect=TimeoutError("Timed out")):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 504
            assert "Timeout" in exc_info.value.detail

    def test_ollama_socket_timeout_raises_504(self):
        provider = OllamaProvider()
        with patch("urllib.request.urlopen", side_effect=socket.timeout("Socket timed out")):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 504

    def test_ollama_url_error_timeout_raises_504(self):
        provider = OllamaProvider()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection timed out")):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 504

    def test_ollama_connection_refused_raises_502(self):
        provider = OllamaProvider()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 502
            assert "Não foi possível conectar ao Ollama" in exc_info.value.detail

    def test_ollama_http_error_raises_502(self):
        provider = OllamaProvider()
        err = urllib.error.HTTPError(
            url="http://localhost:11434/api/generate",
            code=500,
            msg="Internal Error",
            hdrs={},
            fp=io.BytesIO(b"server error"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 502
            assert "500" in exc_info.value.detail

    def test_ollama_invalid_json_raises_502(self):
        provider = OllamaProvider()
        mock_cm = MagicMock()
        mock_cm.read.return_value = b"<html>Not json</html>"
        mock_cm.__enter__.return_value = mock_cm

        with patch("urllib.request.urlopen", return_value=mock_cm):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 502
            assert "não-JSON" in exc_info.value.detail


class TestGrokProvider:
    """Unit tests for the cloud Grok (xAI) provider."""

    def test_grok_missing_api_key_raises_422(self):
        settings = Settings(XAI_API_KEY=None)
        provider = GrokProvider(settings=settings)
        with pytest.raises(HTTPException) as exc_info:
            provider.execute(prompt="Hello Grok")
        assert exc_info.value.status_code == 422
        assert "XAI_API_KEY não configurada" in exc_info.value.detail

    def test_grok_successful_execution(self):
        settings = Settings(
            XAI_API_KEY="test-xai-key-secret",
            DEFAULT_GROK_MODEL="grok-beta",
        )
        provider = GrokProvider(settings=settings)

        mock_data = {
            "choices": [
                {
                    "message": {
                        "content": "Solução arquitetural do Grok.",
                    }
                }
            ],
            "usage": {
                "total_tokens": 120,
            },
        }
        mock_bytes = json.dumps(mock_data).encode("utf-8")

        mock_cm = MagicMock()
        mock_cm.read.return_value = mock_bytes
        mock_cm.__enter__.return_value = mock_cm

        with patch("urllib.request.urlopen", return_value=mock_cm) as mock_urlopen:
            result = provider.execute(prompt="Análise de concorrência", model="grok-2", temperature=0.3)

            assert result["response_text"] == "Solução arquitetural do Grok."
            assert result["model"] == "grok-2"
            assert result["tokens_used"] == 120

            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "https://api.x.ai/v1/chat/completions"
            assert req.headers["Authorization"] == "Bearer test-xai-key-secret"
            sent_body = json.loads(req.data.decode("utf-8"))
            assert sent_body["model"] == "grok-2"
            assert sent_body["temperature"] == 0.3
            assert sent_body["messages"][0]["content"] == "Análise de concorrência"

    def test_grok_unauthorized_401_raises_502_sanitized(self):
        settings = Settings(XAI_API_KEY="invalid-key")
        provider = GrokProvider(settings=settings)

        err = urllib.error.HTTPError(
            url="https://api.x.ai/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(b'{"error": "Unauthorized"}'),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 502
            assert "Falha de autenticação" in exc_info.value.detail
            # Must NOT leak the key!
            assert "invalid-key" not in exc_info.value.detail

    def test_grok_timeout_raises_504(self):
        settings = Settings(XAI_API_KEY="valid-key")
        provider = GrokProvider(settings=settings)

        with patch("urllib.request.urlopen", side_effect=TimeoutError("Timed out")):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 504
            assert "Timeout" in exc_info.value.detail

    def test_grok_connection_failure_raises_502(self):
        settings = Settings(XAI_API_KEY="valid-key")
        provider = GrokProvider(settings=settings)

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("DNS resolution failed")):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 502
            assert "Não foi possível conectar" in exc_info.value.detail


class TestLLMExecutionService:
    """Unit tests for the orchestration service."""

    def test_execution_service_routes_to_mock_provider(self):
        mock_provider = MagicMock()
        mock_provider.execute.return_value = {
            "response_text": "Mocked LLM answer",
            "model": "custom-model",
            "tokens_used": 99,
            "execution_time_seconds": 1.25,
        }

        service = LLMExecutionService()
        service.set_provider(LLMProvider.OLLAMA, mock_provider)

        request = LLMExecutionRequest(
            provider=LLMProvider.OLLAMA,
            model="custom-model",
            temperature=0.8,
        )

        response = service.execute(
            job_id="test-job-uuid",
            prompt="Compile this prompt",
            request=request,
        )

        assert response.job_id == "test-job-uuid"
        assert response.provider == LLMProvider.OLLAMA
        assert response.model == "custom-model"
        assert response.prompt_executed == "Compile this prompt"
        assert response.response_text == "Mocked LLM answer"
        assert response.tokens_used == 99
        assert response.execution_time_seconds == 1.25
