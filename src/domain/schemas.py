"""Domain schemas and DTOs using Pydantic v2 with strict validation."""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.domain.enums import LLMProvider, OutputMode, PromptTechnique, UserRole


class StrictBaseModel(BaseModel):
    """Base model enforcing strict schema validation and forbidding extra fields."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TranscriptionSegment(StrictBaseModel):
    """A segment of transcribed audio with timing information."""

    id: int = Field(..., description="Segment sequential index")
    start: float = Field(..., ge=0.0, description="Start time in seconds")
    end: float = Field(..., ge=0.0, description="End time in seconds")
    text: str = Field(..., description="Segment text content")


class TranscriptionMetadata(StrictBaseModel):
    """Metadata of the transcription process and audio characteristics."""

    duration_seconds: float = Field(..., ge=0.0, description="Audio duration in seconds")
    language: str = Field(..., description="Detected or specified language code (e.g., pt, en)")
    language_probability: float = Field(..., ge=0.0, le=1.0, description="Confidence of detected language")
    model_used: str = Field(..., description="Whisper model size/identifier used")
    compute_type: str = Field(..., description="Quantization compute type (e.g., int8, float16)")
    word_count: int = Field(..., ge=0, description="Total word count of transcription")


class TranscriptionResponse(StrictBaseModel):
    """Detailed response for a completed transcription and prompt engineering job."""

    id: str = Field(..., description="Unique UUID identifier for this transcription job")
    tenant_id: str = Field(..., description="Tenant identifier that owns this job")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    filename: str = Field(..., description="Sanitized original filename")
    file_size_bytes: int = Field(..., ge=0, description="Audio file size in bytes")
    output_mode: OutputMode = Field(..., description="Output mode requested (txt or prompt)")
    prompt_technique: Optional[PromptTechnique] = Field(
        default=None, description="Prompt technique applied if output_mode is prompt"
    )
    raw_text: str = Field(..., description="Direct verbatim transcription from speech-to-text")
    processed_text: str = Field(..., description="Normalized and cleaned transcription via NLP")
    prompt_result: Optional[str] = Field(
        default=None, description="Formatted prompt output ready for LLMs"
    )
    metadata: TranscriptionMetadata = Field(..., description="Execution metadata")


class TranscriptionSummaryResponse(StrictBaseModel):
    """Summarized item for job listing."""

    id: str = Field(..., description="Job UUID")
    tenant_id: str = Field(..., description="Owner tenant ID")
    created_at: str = Field(..., description="ISO 8601 timestamp")
    filename: str = Field(..., description="Sanitized filename")
    output_mode: OutputMode = Field(..., description="Output mode")
    prompt_technique: Optional[PromptTechnique] = Field(default=None)
    duration_seconds: float = Field(..., ge=0.0)
    word_count: int = Field(..., ge=0)


class TranscriptionListResponse(StrictBaseModel):
    """Paginated list of transcriptions for the authenticated tenant."""

    items: List[TranscriptionSummaryResponse] = Field(default_factory=list)
    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)


class PromptTechniqueInfo(StrictBaseModel):
    """Educational metadata explaining a prompt engineering technique."""

    technique: PromptTechnique = Field(..., description="Technique identifier")
    name: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Detailed explanation of the technique")
    when_to_use: str = Field(..., description="Recommended engineering use cases")
    example_structure: str = Field(..., description="Structural pattern of the prompt template")


class TechniquesCatalogResponse(StrictBaseModel):
    """Catalog of all supported prompt engineering techniques."""

    techniques: List[PromptTechniqueInfo] = Field(..., description="List of 6 techniques")


class HealthResponse(StrictBaseModel):
    """Health check response."""

    status: str = Field("healthy")
    service: str = Field("audio-prompt-engine")
    version: str = Field("0.1.0")
    whisper_model: str = Field(...)


class ErrorDetail(StrictBaseModel):
    """Structured error payload."""

    error: str = Field(..., description="Error summary message")
    code: str = Field(..., description="Internal machine-readable error code")
    detail: Optional[str] = Field(default=None, description="Detailed explanation")


class LLMExecutionRequest(StrictBaseModel):
    """Request schema for executing a compiled prompt against an LLM provider."""

    provider: LLMProvider = Field(
        default=LLMProvider.OLLAMA, description="LLM inference provider (ollama or grok)"
    )
    model: Optional[str] = Field(
        default=None, description="Specific model name (e.g. llama3, grok-beta)"
    )
    temperature: Optional[float] = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )


class LLMExecutionResponse(StrictBaseModel):
    """Response schema for an executed LLM prompt."""

    job_id: str = Field(..., description="UUID of the transcription job")
    provider: LLMProvider = Field(..., description="LLM provider used")
    model: str = Field(..., description="Model name executed")
    prompt_executed: str = Field(..., description="The prompt sent to the LLM")
    response_text: str = Field(..., description="The text generated by the LLM")
    execution_time_seconds: float = Field(..., ge=0.0, description="Execution duration in seconds")
    tokens_used: Optional[int] = Field(default=None, ge=0, description="Total tokens consumed if reported")

