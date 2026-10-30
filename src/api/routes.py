"""REST API route handlers for audio-prompt-engine."""

import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)

from src.config import Settings, get_settings
from src.domain.enums import OutputMode, PromptTechnique
from src.domain.schemas import (
    HealthResponse,
    LLMExecutionRequest,
    LLMExecutionResponse,
    TechniquesCatalogResponse,
    TranscriptionListResponse,
    TranscriptionResponse,
)
from src.repositories.transcription_repository import (
    TranscriptionRecord,
    TranscriptionRepository,
    get_transcription_repository,
)
from src.security.auth import (
    AuthContext,
    get_current_auth,
    require_operator,
)
from src.security.sanitizer import (
    AudioValidationError,
    detect_and_validate_audio_format,
    sanitize_filename,
    sanitize_text_input,
)
from src.services.nlp_service import NlpTextNormalizer, get_nlp_service
from src.services.prompt_service import (
    PromptEngineeringService,
    get_prompt_service,
)
from src.services.whisper_service import (
    WhisperTranscriptionService,
    get_whisper_service,
)
from src.services.llm_service import (
    LLMExecutionService,
    get_llm_service,
)

router = APIRouter(prefix="/api/v1", tags=["transcriptions"])

STREAM_CHUNK_SIZE = 64 * 1024  # 64 KB streaming chunks to prevent memory spikes


@router.post(
    "/transcriptions",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload audio and generate transcription or structured prompt",
)
async def create_transcription(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC, M4A)"),
    output_mode: OutputMode = Form(OutputMode.TXT, description="txt or prompt"),
    prompt_technique: Optional[PromptTechnique] = Form(
        None, description="Prompt technique if output_mode=prompt"
    ),
    language: Optional[str] = Form("pt", description="Spoken language code"),
    remove_fillers: bool = Form(True, description="Remove vocal fillers via NLP"),
    custom_context: Optional[str] = Form(None, description="Optional business context (max 2000 chars)"),
    target_audience: Optional[str] = Form(None, description="Target audience (max 200 chars)"),
    auth: AuthContext = Depends(require_operator),
    settings: Settings = Depends(get_settings),
    whisper_service: WhisperTranscriptionService = Depends(get_whisper_service),
    nlp_service: NlpTextNormalizer = Depends(get_nlp_service),
    prompt_service: PromptEngineeringService = Depends(get_prompt_service),
    repository: TranscriptionRepository = Depends(get_transcription_repository),
) -> TranscriptionResponse:
    """Upload audio, validate binary magic bytes, transcribe via Whisper, and apply NLP or Prompt Engineering.

    Requires role 'operator' or 'admin'.
    """
    # ADR-001 contractual adherence: prompt mode requires an explicit technique
    if output_mode == OutputMode.PROMPT and not prompt_technique:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="prompt_technique é obrigatório quando output_mode='prompt'",
        )

    # Validate string constraints
    if custom_context and len(custom_context) > 2000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Field 'custom_context' must not exceed 2000 characters",
        )
    if target_audience and len(target_audience) > 200:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Field 'target_audience' must not exceed 200 characters",
        )

    # Step 1: Read the first chunk to validate empty files and magic bytes before spooling to disk
    try:
        first_chunk = await file.read(STREAM_CHUNK_SIZE)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if not first_chunk or len(first_chunk) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty (0 bytes).",
        )

    try:
        detected_ext = detect_and_validate_audio_format(first_chunk[:32])
    except AudioValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    total_bytes = len(first_chunk)
    if total_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Arquivo excede o limite máximo permitido de {settings.MAX_UPLOAD_SIZE_BYTES} bytes",
        )

    # Step 2: Stream remaining chunks directly into an ephemeral file
    job_id = str(uuid.uuid4())
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{job_id}.{detected_ext}")
    temp_path = temp_file.name
    temp_file.close()

    try:
        with open(temp_path, "wb") as f_out:
            f_out.write(first_chunk)
            while chunk := await file.read(STREAM_CHUNK_SIZE):
                total_bytes += len(chunk)
                if total_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Arquivo excede o limite máximo permitido de {settings.MAX_UPLOAD_SIZE_BYTES} bytes",
                    )
                f_out.write(chunk)

        # Sanitize metadata inputs
        safe_filename = sanitize_filename(file.filename)
        safe_context = sanitize_text_input(custom_context)
        safe_audience = sanitize_text_input(target_audience)

        # Step 3: Transcribe via Whisper
        transcription_result = whisper_service.transcribe_audio(
            temp_path, language=language
        )

        # Step 4: NLP post-processing
        processed_text = nlp_service.clean_text(
            raw_text=transcription_result.raw_text,
            remove_fillers=remove_fillers,
            language=language,
        )

        # Step 5: Prompt Engineering (with Acoustic Prompt Injection mitigation)
        prompt_output: Optional[str] = None
        if output_mode == OutputMode.PROMPT and prompt_technique:
            # Sanitize transcribed speech before interpolating into prompt template
            safe_speech = sanitize_text_input(processed_text) or ""
            prompt_output = prompt_service.generate_prompt(
                technique=prompt_technique,
                processed_text=safe_speech,
                custom_context=safe_context,
                target_audience=safe_audience,
            )

        # Step 6: Persist to repository with strict tenant isolation
        created_at_iso = datetime.now(timezone.utc).isoformat()
        word_count = len(processed_text.split()) if processed_text else 0

        record = TranscriptionRecord(
            id=job_id,
            tenant_id=auth.tenant_id,
            created_at=created_at_iso,
            filename=safe_filename,
            file_size_bytes=total_bytes,
            output_mode=output_mode,
            prompt_technique=prompt_technique if output_mode == OutputMode.PROMPT else None,
            raw_text=transcription_result.raw_text,
            processed_text=processed_text,
            prompt_result=prompt_output,
            duration_seconds=transcription_result.duration_seconds,
            language=transcription_result.language,
            language_probability=transcription_result.language_probability,
            model_used=transcription_result.model_used,
            compute_type=transcription_result.compute_type,
            word_count=word_count,
        )

        repository.save(record)
        return record.to_response()

    finally:
        # Guarantee ephemeral file cleanup
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@router.get(
    "/transcriptions",
    response_model=TranscriptionListResponse,
    summary="List transcriptions for the authenticated tenant",
)
async def list_transcriptions(
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Items offset"),
    auth: AuthContext = Depends(get_current_auth),
    repository: TranscriptionRepository = Depends(get_transcription_repository),
) -> TranscriptionListResponse:
    """Retrieve history of transcriptions strictly filtered by the authenticated tenant."""
    items = repository.list_by_tenant(tenant_id=auth.tenant_id, limit=limit, offset=offset)
    total = repository.count_by_tenant(tenant_id=auth.tenant_id)
    return TranscriptionListResponse(
        items=[item.to_summary() for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/transcriptions/{job_id}",
    response_model=TranscriptionResponse,
    summary="Get details of a transcription by ID",
)
async def get_transcription_by_id(
    job_id: str,
    auth: AuthContext = Depends(get_current_auth),
    repository: TranscriptionRepository = Depends(get_transcription_repository),
) -> TranscriptionResponse:
    """Retrieve details for a single job strictly filtered by tenant (Anti-IDOR and UUID validation)."""
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription not found",
        )

    record = repository.get_by_id(job_id=job_id, tenant_id=auth.tenant_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription not found",
        )
    return record.to_response()


@router.get(
    "/transcriptions/{job_id}/txt",
    summary="Download or view normalized transcription as text/plain",
)
async def get_transcription_txt(
    job_id: str,
    auth: AuthContext = Depends(get_current_auth),
    repository: TranscriptionRepository = Depends(get_transcription_repository),
) -> Response:
    """Return raw text/plain; charset=utf-8 format of the normalized transcription."""
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription not found",
        )

    record = repository.get_by_id(job_id=job_id, tenant_id=auth.tenant_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription not found",
        )

    headers = {
        "Content-Disposition": f'attachment; filename="{job_id}.txt"',
    }
    return Response(
        content=record.processed_text,
        media_type="text/plain; charset=utf-8",
        headers=headers,
    )


@router.post(
    "/transcriptions/{job_id}/execute",
    response_model=LLMExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute a compiled prompt against an LLM provider (Ollama or Grok)",
)
async def execute_prompt_llm(
    job_id: str,
    request: LLMExecutionRequest,
    auth: AuthContext = Depends(require_operator),
    repository: TranscriptionRepository = Depends(get_transcription_repository),
    llm_service: LLMExecutionService = Depends(get_llm_service),
) -> LLMExecutionResponse:
    """Execute a compiled prompt from a transcription job against an LLM provider.

    Requires role 'operator' or 'admin'.
    """
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription not found",
        )

    record = repository.get_by_id(job_id=job_id, tenant_id=auth.tenant_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription not found",
        )

    if record.output_mode != OutputMode.PROMPT or not record.prompt_result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job does not contain a compiled prompt to execute. Only jobs created with output_mode='prompt' can be executed.",
        )

    return llm_service.execute(
        job_id=job_id,
        prompt=record.prompt_result,
        request=request,
    )



@router.get(
    "/techniques",
    response_model=TechniquesCatalogResponse,
    summary="Get catalog of the 6 prompt engineering techniques",
)
async def get_techniques_catalog(
    prompt_service: PromptEngineeringService = Depends(get_prompt_service),
) -> TechniquesCatalogResponse:
    """Return educational metadata about the 6 supported prompt engineering techniques."""
    return TechniquesCatalogResponse(techniques=prompt_service.get_catalog())


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health check endpoint",
)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Verify application health and Whisper engine configuration."""
    return HealthResponse(
        status="healthy",
        service="audio-prompt-engine",
        version="0.1.0",
        whisper_model=f"faster-whisper-{settings.WHISPER_MODEL_SIZE.value}",
    )
