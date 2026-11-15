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
    GeminiProvider,
    GrokProvider,
    GroqProvider,
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
            assert sent_body["options"]["num_predict"] == 512

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

class TestGeminiProvider:
    """Unit tests for the Google AI Studio Gemini provider."""

    def test_gemini_missing_api_key_raises_422(self):
        settings = Settings(GEMINI_API_KEY=None)
        provider = GeminiProvider(settings=settings)
        with pytest.raises(HTTPException) as exc_info:
            provider.execute(prompt="Hello Gemini")
        assert exc_info.value.status_code == 422
        assert "GEMINI_API_KEY não configurada" in exc_info.value.detail

    def test_gemini_successful_execution(self):
        settings = Settings(
            GEMINI_API_KEY="test-gemini-key-secret",
            DEFAULT_GEMINI_MODEL="gemini-1.5-flash",
        )
        provider = GeminiProvider(settings=settings)

        mock_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Resposta rápida do Gemini 1.5 Flash."}
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "totalTokenCount": 85,
            },
        }
        mock_bytes = json.dumps(mock_data).encode("utf-8")

        mock_cm = MagicMock()
        mock_cm.read.return_value = mock_bytes
        mock_cm.__enter__.return_value = mock_cm

        with patch("urllib.request.urlopen", return_value=mock_cm) as mock_urlopen:
            result = provider.execute(prompt="Explique recursão", model="gemini-1.5-pro", temperature=0.2)

            assert result["response_text"] == "Resposta rápida do Gemini 1.5 Flash."
            assert result["model"] == "gemini-1.5-pro"
            assert result["tokens_used"] == 85

            req = mock_urlopen.call_args[0][0]
            assert "gemini-1.5-pro:generateContent" in req.full_url
            assert "key=test-gemini-key-secret" in req.full_url
            sent_body = json.loads(req.data.decode("utf-8"))
            assert sent_body["contents"][0]["parts"][0]["text"] == "Explique recursão"
            assert sent_body["generationConfig"]["temperature"] == 0.2

    def test_gemini_auth_error_raises_502_sanitized(self):
        settings = Settings(GEMINI_API_KEY="invalid-key")
        provider = GeminiProvider(settings=settings)

        err = urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=invalid-key",
            code=400,
            msg="API key not valid",
            hdrs={},
            fp=io.BytesIO(b'{"error": {"message": "API key not valid"}}'),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 502
            assert "Falha de autenticação ou acesso" in exc_info.value.detail
            # Must NOT leak the key!
            assert "invalid-key" not in exc_info.value.detail

    def test_gemini_timeout_raises_504(self):
        settings = Settings(GEMINI_API_KEY="valid-key")
        provider = GeminiProvider(settings=settings)

        with patch("urllib.request.urlopen", side_effect=TimeoutError("Timed out")):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 504
            assert "Timeout" in exc_info.value.detail

    def test_gemini_connection_failure_raises_502(self):
        settings = Settings(GEMINI_API_KEY="valid-key")
        provider = GeminiProvider(settings=settings)

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("DNS resolution failed")):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 502
            assert "Não foi possível conectar" in exc_info.value.detail


class TestGroqProvider:
    """Unit tests for the Groq Cloud LPU provider."""

    def test_groq_missing_api_key_raises_422(self):
        settings = Settings(GROQ_API_KEY=None)
        provider = GroqProvider(settings=settings)
        with pytest.raises(HTTPException) as exc_info:
            provider.execute(prompt="Hello Groq")
        assert exc_info.value.status_code == 422
        assert "GROQ_API_KEY não configurada" in exc_info.value.detail

    def test_groq_successful_execution(self):
        settings = Settings(
            GROQ_API_KEY="test-groq-key-secret",
            DEFAULT_GROQ_MODEL="llama-3.3-70b-versatile",
        )
        provider = GroqProvider(settings=settings)

        mock_data = {
            "choices": [
                {
                    "message": {
                        "content": "Inferência de altíssima velocidade no Groq LPU.",
                    }
                }
            ],
            "usage": {
                "total_tokens": 110,
            },
        }
        mock_bytes = json.dumps(mock_data).encode("utf-8")

        mock_cm = MagicMock()
        mock_cm.read.return_value = mock_bytes
        mock_cm.__enter__.return_value = mock_cm

        with patch("urllib.request.urlopen", return_value=mock_cm) as mock_urlopen:
            result = provider.execute(prompt="Gere uma query SQL", model="llama-3.3-70b-versatile", temperature=0.1)

            assert result["response_text"] == "Inferência de altíssima velocidade no Groq LPU."
            assert result["model"] == "llama-3.3-70b-versatile"
            assert result["tokens_used"] == 110

            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "https://api.groq.com/openai/v1/chat/completions"
            assert req.headers["Authorization"] == "Bearer test-groq-key-secret"
            sent_body = json.loads(req.data.decode("utf-8"))
            assert sent_body["model"] == "llama-3.3-70b-versatile"
            assert sent_body["temperature"] == 0.1
            assert sent_body["messages"][0]["content"] == "Gere uma query SQL"

    def test_groq_unauthorized_401_raises_502_sanitized(self):
        settings = Settings(GROQ_API_KEY="invalid-groq-key")
        provider = GroqProvider(settings=settings)

        err = urllib.error.HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
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
            assert "invalid-groq-key" not in exc_info.value.detail

    def test_groq_timeout_raises_504(self):
        settings = Settings(GROQ_API_KEY="valid-key")
        provider = GroqProvider(settings=settings)

        with patch("urllib.request.urlopen", side_effect=TimeoutError("Timed out")):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 504
            assert "Timeout" in exc_info.value.detail

    def test_groq_connection_failure_raises_502(self):
        settings = Settings(GROQ_API_KEY="valid-key")
        provider = GroqProvider(settings=settings)

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("DNS resolution failed")):
            with pytest.raises(HTTPException) as exc_info:
                provider.execute(prompt="Hello")
            assert exc_info.value.status_code == 502
            assert "Não foi possível conectar" in exc_info.value.detail

    def test_groq_dynamic_api_key_used_when_settings_has_none(self):
        settings = Settings(GROQ_API_KEY=None)
        provider = GroqProvider(settings=settings)

        mock_data = {
            "choices": [{"message": {"content": "Resposta usando chave dinâmica."}}],
            "usage": {"total_tokens": 42},
        }
        mock_bytes = json.dumps(mock_data).encode("utf-8")
        mock_cm = MagicMock()
        mock_cm.read.return_value = mock_bytes
        mock_cm.__enter__.return_value = mock_cm

        with patch("urllib.request.urlopen", return_value=mock_cm) as mock_urlopen:
            result = provider.execute(prompt="Teste", api_key="dynamic-gsk-key")

            assert result["response_text"] == "Resposta usando chave dinâmica."
            req = mock_urlopen.call_args[0][0]
            assert req.headers["Authorization"] == "Bearer dynamic-gsk-key"



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
        assert response.execution_time_seconds == 1.25

    def test_execution_service_routes_to_gemini_provider(self):
        mock_provider = MagicMock()
        mock_provider.execute.return_value = {
            "response_text": "Gemini answer",
            "model": "gemini-1.5-flash",
            "tokens_used": 50,
            "execution_time_seconds": 0.8,
        }

        service = LLMExecutionService()
        service.set_provider(LLMProvider.GEMINI, mock_provider)

        request = LLMExecutionRequest(
            provider=LLMProvider.GEMINI,
            model="gemini-1.5-flash",
            temperature=0.4,
        )

        response = service.execute(
            job_id="test-job-gemini",
            prompt="Analyze this",
            request=request,
        )

        assert response.job_id == "test-job-gemini"
        assert response.provider == LLMProvider.GEMINI
        assert response.model == "gemini-1.5-flash"
        assert response.response_text == "Gemini answer"
        assert response.tokens_used == 50

    def test_execution_service_routes_to_groq_provider(self):
        mock_provider = MagicMock()
        mock_provider.execute.return_value = {
            "response_text": "Groq answer",
            "model": "llama-3.3-70b-versatile",
            "tokens_used": 120,
            "execution_time_seconds": 0.3,
        }

        service = LLMExecutionService()
        service.set_provider(LLMProvider.GROQ, mock_provider)

        request = LLMExecutionRequest(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
        )

        response = service.execute(
            job_id="test-job-groq",
            prompt="Write fast SQL",
            request=request,
        )

        assert response.job_id == "test-job-groq"
        assert response.provider == LLMProvider.GROQ
        assert response.model == "llama-3.3-70b-versatile"
        assert response.response_text == "Groq answer"
        assert response.tokens_used == 120

    def test_execution_service_passes_dynamic_api_key_and_prompt_override(self):
        mock_provider = MagicMock()
        mock_provider.execute.return_value = {
            "response_text": "Answer from dynamic prompt",
            "model": "llama-3.3-70b-versatile",
            "tokens_used": 75,
            "execution_time_seconds": 0.45,
        }

        service = LLMExecutionService()
        service.set_provider(LLMProvider.GROQ, mock_provider)

        request = LLMExecutionRequest(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            api_key="client-supplied-key",
            prompt_override="This is the overridden prompt from the UI",
        )

        response = service.execute(
            job_id="test-job-dynamic",
            prompt="Original compiled prompt",
            request=request,
        )

        assert response.job_id == "test-job-dynamic"
        assert response.prompt_executed == "This is the overridden prompt from the UI"
        assert response.response_text == "Answer from dynamic prompt"

        mock_provider.execute.assert_called_once_with(
            prompt="This is the overridden prompt from the UI",
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            api_key="client-supplied-key",
        )


