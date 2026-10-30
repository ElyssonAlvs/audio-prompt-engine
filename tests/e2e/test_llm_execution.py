"""End-to-End integration tests for LLM execution engine endpoint (POST /api/v1/transcriptions/{job_id}/execute)."""

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.routes import get_llm_service
from src.config import Settings
from src.main import app
from src.services.llm_service import GrokProvider, LLMExecutionService, OllamaProvider


class TestLLMExecutionEndpoint:
    """E2E suite verifying the /execute endpoint against the Pentagon of Defense and ADR specifications."""

    def _create_prompt_job(
        self, client: TestClient, synthetic_wav_bytes: bytes, headers: dict
    ) -> str:
        """Helper to create a valid prompt job."""
        files = {"file": ("prompt.wav", synthetic_wav_bytes, "audio/wav")}
        data = {
            "output_mode": "prompt",
            "prompt_technique": "zero_shot",
            "custom_context": "Contexto técnico de validação",
        }
        resp = client.post("/api/v1/transcriptions", files=files, data=data, headers=headers)
        assert resp.status_code == 201
        return resp.json()["id"]

    def _create_txt_job(
        self, client: TestClient, synthetic_wav_bytes: bytes, headers: dict
    ) -> str:
        """Helper to create a txt-only job (without prompt_result)."""
        files = {"file": ("audio.wav", synthetic_wav_bytes, "audio/wav")}
        data = {"output_mode": "txt"}
        resp = client.post("/api/v1/transcriptions", files=files, data=data, headers=headers)
        assert resp.status_code == 201
        return resp.json()["id"]

    def test_execute_prompt_ollama_200_ok(
        self, client: TestClient, synthetic_wav_bytes: bytes, headers_operator_alpha: dict
    ):
        """Happy path: successfully executes prompt against Ollama local provider."""
        job_id = self._create_prompt_job(client, synthetic_wav_bytes, headers_operator_alpha)

        mock_data = {
            "model": "llama3",
            "response": "Resposta gerada com sucesso pelo Ollama Local.",
            "eval_count": 80,
            "prompt_eval_count": 20,
        }
        mock_cm = MagicMock()
        mock_cm.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_cm.__enter__.return_value = mock_cm

        with patch("urllib.request.urlopen", return_value=mock_cm):
            resp = client.post(
                f"/api/v1/transcriptions/{job_id}/execute",
                json={"provider": "ollama", "model": "llama3", "temperature": 0.7},
                headers=headers_operator_alpha,
            )

            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["job_id"] == job_id
            assert body["provider"] == "ollama"
            assert body["model"] == "llama3"
            assert "ZERO-SHOT" in body["prompt_executed"]
            assert body["response_text"] == "Resposta gerada com sucesso pelo Ollama Local."
            assert body["tokens_used"] == 100
            assert body["execution_time_seconds"] >= 0.0

    def test_execute_prompt_grok_200_ok(
        self, client: TestClient, synthetic_wav_bytes: bytes, headers_operator_alpha: dict
    ):
        """Happy path: successfully executes prompt against xAI Grok Cloud API."""
        job_id = self._create_prompt_job(client, synthetic_wav_bytes, headers_operator_alpha)

        # Inject settings with XAI_API_KEY for this test
        mock_settings = Settings(XAI_API_KEY="test-xai-key-secret", DEFAULT_GROK_MODEL="grok-beta")
        grok_provider = GrokProvider(settings=mock_settings)
        llm_service = LLMExecutionService(settings=mock_settings, grok_provider=grok_provider)
        app.dependency_overrides[get_llm_service] = lambda: llm_service

        mock_data = {
            "choices": [{"message": {"content": "Resposta analítica do Grok."}}],
            "usage": {"total_tokens": 150},
        }
        mock_cm = MagicMock()
        mock_cm.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_cm.__enter__.return_value = mock_cm

        try:
            with patch("urllib.request.urlopen", return_value=mock_cm):
                resp = client.post(
                    f"/api/v1/transcriptions/{job_id}/execute",
                    json={"provider": "grok", "model": "grok-beta", "temperature": 0.5},
                    headers=headers_operator_alpha,
                )

                assert resp.status_code == 200, resp.text
                body = resp.json()
                assert body["job_id"] == job_id
                assert body["provider"] == "grok"
                assert body["model"] == "grok-beta"
                assert body["response_text"] == "Resposta analítica do Grok."
                assert body["tokens_used"] == 150
        finally:
            app.dependency_overrides.pop(get_llm_service, None)

    def test_execute_prompt_404_anti_idor(
        self,
        client: TestClient,
        synthetic_wav_bytes: bytes,
        headers_operator_alpha: dict,
        headers_operator_beta: dict,
    ):
        """Tenant Beta MUST receive 404 Not Found when trying to execute Tenant Alpha's job."""
        alpha_job_id = self._create_prompt_job(client, synthetic_wav_bytes, headers_operator_alpha)

        resp = client.post(
            f"/api/v1/transcriptions/{alpha_job_id}/execute",
            json={"provider": "ollama"},
            headers=headers_operator_beta,
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_execute_prompt_404_invalid_uuid(
        self, client: TestClient, headers_operator_alpha: dict
    ):
        """Invalid UUID string in path returns 404 immediately without database access."""
        resp = client.post(
            "/api/v1/transcriptions/invalid-uuid-string/execute",
            json={"provider": "ollama"},
            headers=headers_operator_alpha,
        )
        assert resp.status_code == 404

    def test_execute_prompt_403_viewer_forbidden(
        self,
        client: TestClient,
        synthetic_wav_bytes: bytes,
        headers_operator_alpha: dict,
        headers_viewer_alpha: dict,
    ):
        """Role 'viewer' receives 403 Forbidden when attempting to execute an LLM prompt."""
        job_id = self._create_prompt_job(client, synthetic_wav_bytes, headers_operator_alpha)

        resp = client.post(
            f"/api/v1/transcriptions/{job_id}/execute",
            json={"provider": "ollama"},
            headers=headers_viewer_alpha,
        )
        assert resp.status_code == 403
        assert "forbidden" in resp.json()["detail"].lower() or "forbidden" in resp.json()["error"].lower()

    def test_execute_prompt_400_job_without_prompt(
        self, client: TestClient, synthetic_wav_bytes: bytes, headers_operator_alpha: dict
    ):
        """Returns 400 Bad Request when attempting to execute a job created with output_mode='txt'."""
        txt_job_id = self._create_txt_job(client, synthetic_wav_bytes, headers_operator_alpha)

        resp = client.post(
            f"/api/v1/transcriptions/{txt_job_id}/execute",
            json={"provider": "ollama"},
            headers=headers_operator_alpha,
        )
        assert resp.status_code == 400
        assert "output_mode='prompt'" in resp.json()["detail"]

    def test_execute_prompt_422_grok_without_api_key(
        self, client: TestClient, synthetic_wav_bytes: bytes, headers_operator_alpha: dict
    ):
        """Returns 422 Unprocessable Content if Grok provider is requested but XAI_API_KEY is not configured."""
        job_id = self._create_prompt_job(client, synthetic_wav_bytes, headers_operator_alpha)

        # Ensure settings has XAI_API_KEY=None
        mock_settings = Settings(XAI_API_KEY=None)
        grok_provider = GrokProvider(settings=mock_settings)
        llm_service = LLMExecutionService(settings=mock_settings, grok_provider=grok_provider)
        app.dependency_overrides[get_llm_service] = lambda: llm_service

        try:
            resp = client.post(
                f"/api/v1/transcriptions/{job_id}/execute",
                json={"provider": "grok"},
                headers=headers_operator_alpha,
            )
            assert resp.status_code == 422
            assert "XAI_API_KEY não configurada" in resp.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_llm_service, None)

    def test_execute_prompt_502_upstream_connection_refused(
        self, client: TestClient, synthetic_wav_bytes: bytes, headers_operator_alpha: dict
    ):
        """Returns 502 Bad Gateway if Ollama server is offline / connection refused."""
        import urllib.error
        job_id = self._create_prompt_job(client, synthetic_wav_bytes, headers_operator_alpha)

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            resp = client.post(
                f"/api/v1/transcriptions/{job_id}/execute",
                json={"provider": "ollama"},
                headers=headers_operator_alpha,
            )
            assert resp.status_code == 502
            assert "Não foi possível conectar" in resp.json()["detail"]

    def test_execute_prompt_504_upstream_timeout(
        self, client: TestClient, synthetic_wav_bytes: bytes, headers_operator_alpha: dict
    ):
        """Returns 504 Gateway Timeout if upstream provider takes longer than timeout limit."""
        job_id = self._create_prompt_job(client, synthetic_wav_bytes, headers_operator_alpha)

        with patch("urllib.request.urlopen", side_effect=TimeoutError("Timed out")):
            resp = client.post(
                f"/api/v1/transcriptions/{job_id}/execute",
                json={"provider": "ollama"},
                headers=headers_operator_alpha,
            )
            assert resp.status_code == 504
            assert "Timeout" in resp.json()["detail"]
