"""End-to-End integration tests for audio transcription and prompt engineering flows."""

from fastapi.testclient import TestClient


class TestApiFlows:
    """E2E verification of transcription, NLP cleaning, and prompt generation."""

    def test_flow_transcription_txt_mode(
        self, client: TestClient, synthetic_wav_bytes: bytes, headers_operator_alpha: dict
    ):
        # Step 1: Upload WAV file with output_mode=txt
        files = {"file": ("recording.wav", synthetic_wav_bytes, "audio/wav")}
        data = {
            "output_mode": "txt",
            "language": "pt",
            "remove_fillers": "true",
        }
        create_resp = client.post(
            "/api/v1/transcriptions",
            files=files,
            data=data,
            headers=headers_operator_alpha,
        )
        assert create_resp.status_code == 201, create_resp.text
        job_data = create_resp.json()

        job_id = job_data["id"]
        assert job_data["tenant_id"] == "tenant-alpha"
        assert job_data["output_mode"] == "txt"
        assert job_data["prompt_technique"] is None
        assert job_data["prompt_result"] is None
        assert "ééé" not in job_data["processed_text"]
        assert job_data["metadata"]["duration_seconds"] > 0

        # Step 2: Retrieve transcription details by ID
        get_resp = client.get(
            f"/api/v1/transcriptions/{job_id}",
            headers=headers_operator_alpha,
        )
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == job_id
        assert get_resp.json()["processed_text"] == job_data["processed_text"]

        # Step 3: Download text/plain via /txt endpoint
        txt_resp = client.get(
            f"/api/v1/transcriptions/{job_id}/txt",
            headers=headers_operator_alpha,
        )
        assert txt_resp.status_code == 200
        assert "text/plain" in txt_resp.headers["content-type"]
        assert txt_resp.text == job_data["processed_text"]

    def test_flow_prompt_mode_with_cot_technique(
        self, client: TestClient, synthetic_wav_bytes: bytes, headers_operator_alpha: dict
    ):
        files = {"file": ("audio_task.wav", synthetic_wav_bytes, "audio/wav")}
        data = {
            "output_mode": "prompt",
            "prompt_technique": "chain_of_thought",
            "custom_context": "Arquitetura de microsserviços distribuídos",
        }
        resp = client.post(
            "/api/v1/transcriptions",
            files=files,
            data=data,
            headers=headers_operator_alpha,
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()

        assert body["output_mode"] == "prompt"
        assert body["prompt_technique"] == "chain_of_thought"
        assert body["prompt_result"] is not None
        assert "<user_speech>" in body["prompt_result"]
        assert "Decomposição do Problema" in body["prompt_result"]
        assert "Arquitetura de microsserviços distribuídos" in body["prompt_result"]

    def test_flow_prompt_mode_with_co_star_technique(
        self, client: TestClient, synthetic_wav_bytes: bytes, headers_operator_alpha: dict
    ):
        files = {"file": ("demand.wav", synthetic_wav_bytes, "audio/wav")}
        data = {
            "output_mode": "prompt",
            "prompt_technique": "co_star",
            "custom_context": "Fintech Core Banking API",
            "target_audience": "Engenheiros de Dados e Arquitetos",
        }
        resp = client.post(
            "/api/v1/transcriptions",
            files=files,
            data=data,
            headers=headers_operator_alpha,
        )
        assert resp.status_code == 201
        body = resp.json()

        assert body["output_mode"] == "prompt"
        assert body["prompt_technique"] == "co_star"
        assert "CO-STAR" in body["prompt_result"]
        assert "Fintech Core Banking API" in body["prompt_result"]
        assert "Engenheiros de Dados e Arquitetos" in body["prompt_result"]

    def test_flow_list_transcriptions(
        self, client: TestClient, synthetic_wav_bytes: bytes, headers_operator_alpha: dict
    ):
        # Create 2 records
        for i in range(2):
            files = {"file": (f"test_{i}.wav", synthetic_wav_bytes, "audio/wav")}
            data = {"output_mode": "txt"}
            client.post(
                "/api/v1/transcriptions",
                files=files,
                data=data,
                headers=headers_operator_alpha,
            )

        list_resp = client.get(
            "/api/v1/transcriptions",
            headers=headers_operator_alpha,
        )
        assert list_resp.status_code == 200
        body = list_resp.json()
        assert body["total"] >= 2
        assert len(body["items"]) >= 2
        assert body["limit"] == 50
        assert body["offset"] == 0

    def test_flow_techniques_catalog(self, client: TestClient):
        resp = client.get("/api/v1/techniques")
        assert resp.status_code == 200
        catalog = resp.json()
        assert "techniques" in catalog
        assert len(catalog["techniques"]) == 6
        names = [t["name"] for t in catalog["techniques"]]
        assert "Chain of Thought (CoT)" in names
        assert "CO-STAR Framework" in names

    def test_flow_health_check(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "audio-prompt-engine"
