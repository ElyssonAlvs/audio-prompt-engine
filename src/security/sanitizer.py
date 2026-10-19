"""Security validators: audio magic bytes, input sanitization, and path traversal protection."""

import html
import re
from pathlib import Path
from typing import Optional


class AudioValidationError(Exception):
    """Raised when uploaded audio fails security or integrity checks."""


def detect_and_validate_audio_format(header: bytes) -> str:
    """Validate audio binary magic bytes and return the detected extension format.

    Supported formats:
    - WAV: 'RIFF' at 0..4 and 'WAVE' at 8..12
    - MP3: 'ID3' at 0..3 or MPEG frame sync (0xFF and second byte 0xE0 mask)
    - OGG: 'OggS' at 0..4
    - FLAC: 'fLaC' at 0..4
    - M4A: 'ftyp' in bytes 4..12
    """
    if len(header) < 12:
        raise AudioValidationError("Audio file header is too short to verify format signature.")

    # WAV
    if header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"WAVE":
        return "wav"

    # MP3 with ID3 tag
    if header[:3] == b"ID3":
        return "mp3"

    # MP3 raw frame sync (0xFFE0 mask)
    if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return "mp3"

    # OGG container
    if header[:4] == b"OggS":
        return "ogg"

    # FLAC
    if header[:4] == b"fLaC":
        return "flac"

    # M4A / AAC container
    if b"ftyp" in header[4:16]:
        return "m4a"

    raise AudioValidationError(
        "Invalid or unsupported audio format. Magic bytes mismatch: expected WAV, MP3, OGG, FLAC, or M4A."
    )


def validate_file_size(size_bytes: int, max_bytes: int) -> None:
    """Validate that the file size is non-zero and within configured limits."""
    if size_bytes <= 0:
        raise AudioValidationError("Audio file is empty (0 bytes).")
    if size_bytes > max_bytes:
        raise AudioValidationError(
            f"Audio file size ({size_bytes} bytes) exceeds maximum allowed limit of {max_bytes} bytes."
        )


def sanitize_filename(filename: Optional[str]) -> str:
    """Sanitize filename to prevent Path Traversal and illegal character exploits."""
    if not filename:
        return "audio_upload.wav"

    # Extract base name to defeat directory traversal (e.g., ../../../etc/passwd)
    base = Path(filename).name

    # Remove any null bytes or path separators
    base = base.replace("\x00", "").replace("/", "").replace("\\", "")

    # Retain only alphanumeric, dots, underscores, and dashes
    clean = re.sub(r"[^a-zA-Z0-9._-]", "_", base)

    # Ensure it's not empty or just dots
    clean = clean.lstrip(".")
    if not clean:
        clean = "audio_upload.wav"

    return clean[:100]


def sanitize_text_input(text: Optional[str]) -> Optional[str]:
    """Sanitize textual input against XSS, control characters, and delimiter injection."""
    if text is None:
        return None

    # Remove null bytes and control chars (preserve newlines and tabs)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Strip dangerous HTML script tags
    cleaned = re.sub(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<\s*script[^>]*>", "", cleaned, flags=re.IGNORECASE)

    # Escape XML delimiters to avoid breaking prompt template boundaries
    cleaned = cleaned.replace("</user_speech>", "&lt;/user_speech&gt;")
    cleaned = cleaned.replace("<user_speech>", "&lt;user_speech&gt;")

    return cleaned.strip()
