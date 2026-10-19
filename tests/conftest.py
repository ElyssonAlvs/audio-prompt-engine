"""Global Pytest fixtures for unit, e2e, and security test suites."""

import io
import wave
from typing import Generator
import pytest
from fastapi.testclient import TestClient

from src.api.routes import (
    get_transcription_repository,
    get_whisper_service,
)
from src.main import app
from src.repositories.transcription_repository import TranscriptionRepository
from src.services.whisper_service import MockWhisperEngine, WhisperTranscriptionService


@pytest.fixture
def synthetic_wav_bytes() -> bytes:
    """Generate a 100% valid in-memory PCM WAV audio file with RIFF headers."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)  # 16kHz
        # 0.5 second of silence / tone
        wf.writeframes(b"\x00\x00" * 8000)
    return buf.getvalue()


@pytest.fixture
def fake_mp3_bytes() -> bytes:
    """Generate valid MP3 header bytes (ID3v2 standard tag)."""
    # ID3 header: 'ID3' + version (0x04 0x00) + flags (0x00) + size (0x00 0x00 0x00 0x0A) + 10 bytes padding
    return b"ID3\x04\x00\x00\x00\x00\x00\x0a" + (b"\x00" * 20)


@pytest.fixture
def test_repo(tmp_path) -> TranscriptionRepository:
    """Provide an isolated, fresh SQLite repository for each test."""
    db_file = tmp_path / "test_audio_engine.db"
    return TranscriptionRepository(db_path=str(db_file))


@pytest.fixture
def mock_whisper_engine() -> MockWhisperEngine:
    """Deterministic mock speech engine."""
    return MockWhisperEngine(
        mock_text="Olá equipe, ééé... nós precisamos tipo assim criar uma rota FastAPI com validação de magic bytes, né? Para proteger o sistema contra uploads maliciosos.",
        duration=5.0,
        language="pt",
    )


@pytest.fixture
def client(test_repo, mock_whisper_engine) -> Generator[TestClient, None, None]:
    """TestClient configured with dependency overrides for isolated fast testing."""
    whisper_service = WhisperTranscriptionService(engine=mock_whisper_engine)

    app.dependency_overrides[get_transcription_repository] = lambda: test_repo
    app.dependency_overrides[get_whisper_service] = lambda: whisper_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def headers_operator_alpha() -> dict:
    return {"X-API-Key": "dev-operator-alpha"}


@pytest.fixture
def headers_viewer_alpha() -> dict:
    return {"X-API-Key": "dev-viewer-alpha"}


@pytest.fixture
def headers_operator_beta() -> dict:
    return {"X-API-Key": "dev-operator-beta"}


@pytest.fixture
def headers_admin() -> dict:
    return {"X-API-Key": "dev-admin"}
