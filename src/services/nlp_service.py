"""Deterministic Natural Language Processing normalization service for text outputs."""

import re
from typing import List, Optional


class NlpTextNormalizer:
    """Deterministic NLP pipeline for cleaning and normalizing transcribed speech."""

    # Portuguese and general speech filler patterns
    FILLER_PATTERNS: List[re.Pattern] = [
        # Elongated hesitation sounds optionally followed by dots
        re.compile(r"\b[eé]{3,}\b\.{0,3}", re.IGNORECASE),
        re.compile(r"\b[aá]{3,}\b\.{0,3}", re.IGNORECASE),
        re.compile(r"\b[ãa]{3,}\b\.{0,3}", re.IGNORECASE),
        re.compile(r"\bhu+m+\b\.{0,3}", re.IGNORECASE),
        re.compile(r"\bu+h+m*\b\.{0,3}", re.IGNORECASE),
        re.compile(r"\be+r+r*\b\.{0,3}", re.IGNORECASE),
        # Common verbal disfluencies in PT
        re.compile(r"\btipo assim\b", re.IGNORECASE),
        re.compile(r"\baí ó\b", re.IGNORECASE),
        re.compile(r"\btá ligado\b", re.IGNORECASE),
        re.compile(r"\bpera aí\b", re.IGNORECASE),
        re.compile(r"\bentão né\b", re.IGNORECASE),
        re.compile(r"\bné\b", re.IGNORECASE),
        # Standalone or hesitation ellipses attached to vowel sounds
        re.compile(r"\.{2,}\s*[eé]{2,}", re.IGNORECASE),
        re.compile(r"[eé]{2,}\s*\.{2,}", re.IGNORECASE),
    ]

    @classmethod
    def remove_fillers(cls, text: str) -> str:
        """Remove vocal fillers and speech crutches while preserving semantics."""
        cleaned = text
        for pattern in cls.FILLER_PATTERNS:
            cleaned = pattern.sub("", cleaned)
        return cleaned

    @classmethod
    def normalize_punctuation_and_spaces(cls, text: str) -> str:
        """Fix broken spacing, duplicated punctuation, and orphan commas."""
        cleaned = text

        # Remove spaces before punctuation
        cleaned = re.sub(r"\s+([,.\!?;:])", r"\1", cleaned)

        # Eliminate orphan/repeated punctuation caused by filler removal (e.g. ', ,' or ', ?' or ', .')
        cleaned = re.sub(r",\s*([?!;:.,])", r"\1", cleaned)
        cleaned = re.sub(r",\s*,+", ",", cleaned)
        cleaned = re.sub(r",\s*\.", ".", cleaned)
        cleaned = re.sub(r"\.\s*,", ".", cleaned)
        cleaned = re.sub(r"\?+", "?", cleaned)
        cleaned = re.sub(r"!+", "!", cleaned)
        cleaned = re.sub(r":+", ":", cleaned)
        cleaned = re.sub(r";+", ";", cleaned)

        # Normalize multiple dots: replace '..' with '.' and '....+' with '...'
        cleaned = re.sub(r"(?<!\.)\.\.(?!\.)", ".", cleaned)
        cleaned = re.sub(r"\.{4,}", "...", cleaned)

        # Ensure single space after punctuation when followed by word characters
        cleaned = re.sub(r"([,.\!?;:])([^\s\d,.\!?;:])", r"\1 \2", cleaned)

        # Collapse multiple spaces into one
        cleaned = re.sub(r"[ \t]+", " ", cleaned)

        # Remove leading punctuation (e.g., if sentence began with a comma after filler removal)
        cleaned = re.sub(r"^[\s,;]+", "", cleaned)

        return cleaned.strip()

    @classmethod
    def capitalize_sentences(cls, text: str) -> str:
        """Ensure the first word of each sentence or paragraph is capitalized."""
        if not text:
            return ""

        def capitalize_match(match: re.Match) -> str:
            prefix = match.group(1)
            char = match.group(2)
            return prefix + char.upper()

        # Capitalize after sentence boundaries (. ! ? \n)
        pattern = re.compile(r"(^|[.!?\n]\s*)([a-zà-ú])", re.UNICODE)
        return pattern.sub(capitalize_match, text)

    def clean_text(
        self,
        raw_text: str,
        remove_fillers: bool = True,
        language: Optional[str] = "pt",
    ) -> str:
        """Run the full deterministic NLP pipeline over raw transcription text."""
        if not raw_text or not raw_text.strip():
            return ""

        text = raw_text.strip()

        if remove_fillers:
            text = self.remove_fillers(text)

        text = self.normalize_punctuation_and_spaces(text)
        text = self.capitalize_sentences(text)

        # Ensure final sentence ends with proper punctuation
        if text and text[-1] not in ".!?":
            text += "."

        return text


# Global instance
_nlp_service_instance: Optional[NlpTextNormalizer] = None


def get_nlp_service() -> NlpTextNormalizer:
    global _nlp_service_instance
    if _nlp_service_instance is None:
        _nlp_service_instance = NlpTextNormalizer()
    return _nlp_service_instance
