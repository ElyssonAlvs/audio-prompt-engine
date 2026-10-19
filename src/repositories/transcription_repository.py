"""Multi-tenant parameterized SQLite repository for transcription and prompt storage."""

import sqlite3
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

from src.config import get_settings
from src.domain.enums import OutputMode, PromptTechnique
from src.domain.schemas import (
    TranscriptionMetadata,
    TranscriptionResponse,
    TranscriptionSummaryResponse,
)


class TranscriptionRecord(BaseModel):
    """Entity representing a persisted transcription job in the repository."""

    model_config = ConfigDict(extra="forbid")

    id: str
    tenant_id: str
    created_at: str
    filename: str
    file_size_bytes: int
    output_mode: OutputMode
    prompt_technique: Optional[PromptTechnique] = None
    raw_text: str
    processed_text: str
    prompt_result: Optional[str] = None
    duration_seconds: float
    language: str
    language_probability: float
    model_used: str
    compute_type: str
    word_count: int

    def to_response(self) -> TranscriptionResponse:
        """Convert repository entity to public API response schema."""
        return TranscriptionResponse(
            id=self.id,
            tenant_id=self.tenant_id,
            created_at=self.created_at,
            filename=self.filename,
            file_size_bytes=self.file_size_bytes,
            output_mode=self.output_mode,
            prompt_technique=self.prompt_technique,
            raw_text=self.raw_text,
            processed_text=self.processed_text,
            prompt_result=self.prompt_result,
            metadata=TranscriptionMetadata(
                duration_seconds=self.duration_seconds,
                language=self.language,
                language_probability=self.language_probability,
                model_used=self.model_used,
                compute_type=self.compute_type,
                word_count=self.word_count,
            ),
        )

    def to_summary(self) -> TranscriptionSummaryResponse:
        """Convert repository entity to summary schema."""
        return TranscriptionSummaryResponse(
            id=self.id,
            tenant_id=self.tenant_id,
            created_at=self.created_at,
            filename=self.filename,
            output_mode=self.output_mode,
            prompt_technique=self.prompt_technique,
            duration_seconds=self.duration_seconds,
            word_count=self.word_count,
        )


class TranscriptionRepository:
    """Repository handling SQLite operations with strict multi-tenant isolation and Windows-safe resource cleanup."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_settings().DATABASE_PATH
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema with proper indexes, ensuring connection is closed."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS transcriptions (
                        id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        file_size_bytes INTEGER NOT NULL,
                        output_mode TEXT NOT NULL,
                        prompt_technique TEXT,
                        raw_text TEXT NOT NULL,
                        processed_text TEXT NOT NULL,
                        prompt_result TEXT,
                        duration_seconds REAL NOT NULL,
                        language TEXT NOT NULL,
                        language_probability REAL NOT NULL,
                        model_used TEXT NOT NULL,
                        compute_type TEXT NOT NULL,
                        word_count INTEGER NOT NULL
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tenant_created ON transcriptions (tenant_id, created_at DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_id_tenant ON transcriptions (id, tenant_id)"
                )
        finally:
            conn.close()

    def save(self, record: TranscriptionRecord) -> None:
        """Save a new transcription job record using 100% parameterized query and explicit connection closure."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO transcriptions (
                        id, tenant_id, created_at, filename, file_size_bytes,
                        output_mode, prompt_technique, raw_text, processed_text,
                        prompt_result, duration_seconds, language, language_probability,
                        model_used, compute_type, word_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.tenant_id,
                        record.created_at,
                        record.filename,
                        record.file_size_bytes,
                        record.output_mode.value,
                        record.prompt_technique.value if record.prompt_technique else None,
                        record.raw_text,
                        record.processed_text,
                        record.prompt_result,
                        record.duration_seconds,
                        record.language,
                        record.language_probability,
                        record.model_used,
                        record.compute_type,
                        record.word_count,
                    ),
                )
        finally:
            conn.close()

    def get_by_id(self, job_id: str, tenant_id: str) -> Optional[TranscriptionRecord]:
        """Retrieve transcription strictly enforcing tenant isolation (Anti-IDOR)."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM transcriptions WHERE id = ? AND tenant_id = ?",
                (job_id, tenant_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_record(row)
        finally:
            conn.close()

    def list_by_tenant(
        self, tenant_id: str, limit: int = 50, offset: int = 0
    ) -> List[TranscriptionRecord]:
        """List transcriptions strictly filtered by tenant."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM transcriptions
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (tenant_id, limit, offset),
            )
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            conn.close()

    def count_by_tenant(self, tenant_id: str) -> int:
        """Count total transcriptions strictly for the given tenant."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) as total FROM transcriptions WHERE tenant_id = ?",
                (tenant_id,),
            )
            row = cursor.fetchone()
            return int(row["total"]) if row else 0
        finally:
            conn.close()

    def delete_by_id(self, job_id: str, tenant_id: str) -> bool:
        """Delete transcription strictly isolated by tenant."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM transcriptions WHERE id = ? AND tenant_id = ?",
                    (job_id, tenant_id),
                )
                return cursor.rowcount > 0
        finally:
            conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TranscriptionRecord:
        technique = None
        if row["prompt_technique"]:
            try:
                technique = PromptTechnique(row["prompt_technique"])
            except ValueError:
                technique = None

        return TranscriptionRecord(
            id=row["id"],
            tenant_id=row["tenant_id"],
            created_at=row["created_at"],
            filename=row["filename"],
            file_size_bytes=row["file_size_bytes"],
            output_mode=OutputMode(row["output_mode"]),
            prompt_technique=technique,
            raw_text=row["raw_text"],
            processed_text=row["processed_text"],
            prompt_result=row["prompt_result"],
            duration_seconds=float(row["duration_seconds"]),
            language=row["language"],
            language_probability=float(row["language_probability"]),
            model_used=row["model_used"],
            compute_type=row["compute_type"],
            word_count=int(row["word_count"]),
        )


# Global repository instance
_transcription_repo_instance: Optional[TranscriptionRepository] = None


def get_transcription_repository() -> TranscriptionRepository:
    global _transcription_repo_instance
    if _transcription_repo_instance is None:
        _transcription_repo_instance = TranscriptionRepository()
    return _transcription_repo_instance
