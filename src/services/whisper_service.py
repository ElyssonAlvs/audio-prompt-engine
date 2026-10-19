"""Audio transcription service wrapping faster-whisper with dependency injection."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings, get_settings
from src.domain.schemas import TranscriptionSegment

logger = logging.getLogger(__name__)


class WhisperResult(BaseModel):
    """Normalized output from speech-to-text inference."""

    model_config = ConfigDict(extra="forbid")

    raw_text: str
    segments: List[TranscriptionSegment]
    duration_seconds: float = Field(ge=0.0)
    language: str
    language_probability: float = Field(ge=0.0, le=1.0)
    model_used: str
    compute_type: str


class IWhisperEngine(ABC):
    """Abstract interface for Whisper transcription backends."""

    @abstractmethod
    def transcribe(self, audio_file_path: str, language: Optional[str] = "pt") -> WhisperResult:
        """Transcribe an audio file at the given path."""
        pass


class FasterWhisperEngine(IWhisperEngine):
    """Concrete faster-whisper (CTranslate2) implementation."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            model_size = self.settings.WHISPER_MODEL_SIZE.value
            device = self.settings.WHISPER_DEVICE
            compute_type = self.settings.WHISPER_COMPUTE_TYPE
            logger.info(
                f"Loading faster-whisper model: size={model_size}, device={device}, compute_type={compute_type}"
            )
            self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        return self._model

    def transcribe(self, audio_file_path: str, language: Optional[str] = "pt") -> WhisperResult:
        model = self._get_model()
        segments_gen, info = model.transcribe(
            audio_file_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        segments_list: List[TranscriptionSegment] = []
        raw_chunks: List[str] = []

        for seg in segments_gen:
            segments_list.append(
                TranscriptionSegment(
                    id=seg.id,
                    start=round(float(seg.start), 2),
                    end=round(float(seg.end), 2),
                    text=seg.text.strip(),
                )
            )
            raw_chunks.append(seg.text.strip())

        full_raw_text = " ".join(raw_chunks).strip()
        duration = float(info.duration) if info.duration else (segments_list[-1].end if segments_list else 0.0)

        return WhisperResult(
            raw_text=full_raw_text,
            segments=segments_list,
            duration_seconds=round(duration, 2),
            language=info.language or (language or "unknown"),
            language_probability=round(float(info.language_probability or 1.0), 3),
            model_used=f"faster-whisper-{self.settings.WHISPER_MODEL_SIZE.value}",
            compute_type=self.settings.WHISPER_COMPUTE_TYPE,
        )


class MockWhisperEngine(IWhisperEngine):
    """Deterministic mock Whisper engine for tests and offline development."""

    def __init__(
        self,
        mock_text: Optional[str] = None,
        duration: float = 8.5,
        language: str = "pt",
    ):
        self.mock_text = (
            mock_text
            or "Olá equipe, ééé... nós precisamos tipo assim criar uma rota FastAPI com validação de magic bytes, né? Para proteger o sistema contra uploads maliciosos."
        )
        self.duration = duration
        self.language = language

    def transcribe(self, audio_file_path: str, language: Optional[str] = "pt") -> WhisperResult:
        return WhisperResult(
            raw_text=self.mock_text,
            segments=[
                TranscriptionSegment(
                    id=0,
                    start=0.0,
                    end=round(self.duration / 2, 2),
                    text=self.mock_text[: len(self.mock_text) // 2],
                ),
                TranscriptionSegment(
                    id=1,
                    start=round(self.duration / 2, 2),
                    end=round(self.duration, 2),
                    text=self.mock_text[len(self.mock_text) // 2 :],
                ),
            ],
            duration_seconds=self.duration,
            language=language or self.language,
            language_probability=0.985,
            model_used="mock-whisper-engine",
            compute_type="int8-simulated",
        )


class WhisperTranscriptionService:
    """High-level service coordinating transcription."""

    def __init__(self, engine: Optional[IWhisperEngine] = None):
        self.engine = engine or FasterWhisperEngine()

    def set_engine(self, engine: IWhisperEngine) -> None:
        """Swap backend engine (useful for tests)."""
        self.engine = engine

    def transcribe_audio(self, audio_file_path: str, language: Optional[str] = "pt") -> WhisperResult:
        return self.engine.transcribe(audio_file_path, language=language)


# Global service instance with FasterWhisperEngine by default
_whisper_service_instance: Optional[WhisperTranscriptionService] = None


def get_whisper_service() -> WhisperTranscriptionService:
    global _whisper_service_instance
    if _whisper_service_instance is None:
        _whisper_service_instance = WhisperTranscriptionService(engine=FasterWhisperEngine())
    return _whisper_service_instance
