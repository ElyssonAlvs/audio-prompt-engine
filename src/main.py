"""FastAPI main application entrypoint with security exception handlers and lifecycle management."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import router as api_router
from src.config import get_settings
from src.domain.schemas import HealthResponse
from src.repositories.transcription_repository import get_transcription_repository
from src.security.sanitizer import AudioValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("audio_prompt_engine")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize database schema and resources."""
    logger.info("Initializing audio-prompt-engine services...")
    settings = get_settings()
    # Initialize repository schema
    repo = get_transcription_repository()
    logger.info(f"Database initialized at: {repo.db_path}")
    logger.info(
        f"Whisper configuration: model={settings.WHISPER_MODEL_SIZE.value}, device={settings.WHISPER_DEVICE}, compute_type={settings.WHISPER_COMPUTE_TYPE}"
    )
    yield
    logger.info("Shutting down audio-prompt-engine.")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title="Audio Prompt Engine",
        description=(
            "Motor inteligente de transcrição de áudio com Whisper e síntese de prompts "
            "estruturados para LLMs através de 6 técnicas consagradas de engenharia de prompt."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.DEBUG else ["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Global Exception Handlers (Defense-in-depth: zero stack trace leakage)
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "detail": exc.detail,
                "code": "HTTP_ERROR",
                "status_code": exc.status_code,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Standardize validation error output
        formatted_errors = []
        for err in exc.errors():
            formatted_errors.append(
                {
                    "loc": [str(loc) for loc in err.get("loc", [])],
                    "msg": err.get("msg", "Invalid value"),
                    "type": err.get("type", "value_error"),
                }
            )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "Validation error",
                "code": "VALIDATION_ERROR",
                "details": formatted_errors,
            },
        )

    @app.exception_handler(AudioValidationError)
    async def audio_validation_exception_handler(request: Request, exc: AudioValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": str(exc),
                "code": "AUDIO_VALIDATION_ERROR",
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Log real exception internally without exposing stack trace to the caller
        logger.error(f"Unhandled server error at {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error occurred",
                "code": "INTERNAL_SERVER_ERROR",
            },
        )

    @app.get("/health", response_model=HealthResponse, tags=["system"], summary="Root health check")
    async def root_health_check() -> HealthResponse:
        cfg = get_settings()
        return HealthResponse(
            status="healthy",
            service="audio-prompt-engine",
            version="0.1.0",
            whisper_model=f"faster-whisper-{cfg.WHISPER_MODEL_SIZE.value}",
        )

    # Register routes
    app.include_router(api_router)

    return app


app = create_app()
