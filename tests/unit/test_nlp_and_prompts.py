"""Unit tests for NLP text normalization, prompt engineering strategies, and security sanitizers."""

import pytest

from src.domain.enums import PromptTechnique
from src.security.sanitizer import (
    AudioValidationError,
    detect_and_validate_audio_format,
    sanitize_filename,
    sanitize_text_input,
    validate_file_size,
)
from src.services.nlp_service import NlpTextNormalizer
from src.services.prompt_service import (
    ChainOfThoughtStrategy,
    CoStarStrategy,
    FewShotStrategy,
    PromptEngineeringService,
    ReActStrategy,
    RolePromptingStrategy,
    ZeroShotStrategy,
)


class TestNlpNormalizer:
    """Unit tests for the NLP normalization pipeline."""

    def setup_method(self):
        self.normalizer = NlpTextNormalizer()

    def test_remove_fillers_portuguese(self):
        raw = "Olá equipe, ééé... nós precisamos tipo assim criar uma rota, né?"
        cleaned = self.normalizer.remove_fillers(raw)
        assert "ééé" not in cleaned.lower()
        assert "tipo assim" not in cleaned.lower()
        assert "né" not in cleaned.lower()

    def test_normalize_punctuation_and_spaces(self):
        text = "Olá ,   nós precisamos de  uma rota  ..  Vamos testar   ??"
        normalized = self.normalizer.normalize_punctuation_and_spaces(text)
        assert "Olá," in normalized
        assert "  " not in normalized
        assert ".." not in normalized
        assert "??" not in normalized

    def test_capitalize_sentences(self):
        text = "olá mundo. este é um teste! funcionou bem? claro que sim."
        capitalized = self.normalizer.capitalize_sentences(text)
        assert capitalized.startswith("Olá mundo.")
        assert "Este é um teste!" in capitalized
        assert "Funcionou bem?" in capitalized
        assert "Claro que sim." in capitalized

    def test_full_pipeline_clean_text(self):
        raw = "   olá equipe, ééé... nós precisamos tipo assim criar uma rota FastAPI, né?   para proteger o sistema   "
        result = self.normalizer.clean_text(raw, remove_fillers=True)
        assert result.startswith("Olá equipe,")
        assert "ééé" not in result
        assert "tipo assim" not in result
        assert "né" not in result
        assert "Para proteger o sistema." in result


class TestPromptEngineeringStrategies:
    """Unit tests for all 6 prompt engineering techniques."""

    def setup_method(self):
        self.service = PromptEngineeringService()
        self.text = "Construir um microsserviço de autenticação JWT em Python com FastAPI."

    def test_zero_shot_strategy(self):
        prompt = self.service.generate_prompt(
            technique=PromptTechnique.ZERO_SHOT,
            processed_text=self.text,
            custom_context="Contexto corporativo",
        )
        assert "ZERO-SHOT" in prompt
        assert "<user_speech>" in prompt
        assert "</user_speech>" in prompt
        assert self.text in prompt
        assert "Contexto corporativo" in prompt

    def test_few_shot_strategy(self):
        prompt = self.service.generate_prompt(
            technique=PromptTechnique.FEW_SHOT,
            processed_text=self.text,
        )
        assert "FEW-SHOT" in prompt
        assert "Exemplo 1:" in prompt
        assert "Exemplo 2:" in prompt
        assert "<user_speech>" in prompt
        assert self.text in prompt

    def test_chain_of_thought_strategy(self):
        prompt = self.service.generate_prompt(
            technique=PromptTechnique.CHAIN_OF_THOUGHT,
            processed_text=self.text,
        )
        assert "CHAIN-OF-THOUGHT" in prompt
        assert "Decomposição do Problema" in prompt
        assert "Identificação de Riscos" in prompt
        assert "Raciocínio Dedutivo" in prompt
        assert "<user_speech>" in prompt
        assert self.text in prompt

    def test_role_prompting_strategy(self):
        prompt = self.service.generate_prompt(
            technique=PromptTechnique.ROLE_PROMPTING,
            processed_text=self.text,
        )
        assert "ROLE PROMPTING" in prompt
        assert "Staff Software Engineer" in prompt
        assert "<user_speech>" in prompt
        assert self.text in prompt

    def test_co_star_strategy(self):
        prompt = self.service.generate_prompt(
            technique=PromptTechnique.CO_STAR,
            processed_text=self.text,
            custom_context="Contexto de Segurança Financeira",
            target_audience="Diretores de TI",
        )
        assert "CO-STAR" in prompt
        assert "CONTEXT" in prompt
        assert "OBJECTIVE" in prompt
        assert "STYLE" in prompt
        assert "TONE" in prompt
        assert "AUDIENCE" in prompt
        assert "RESPONSE FORMAT" in prompt
        assert "Contexto de Segurança Financeira" in prompt
        assert "Diretores de TI" in prompt
        assert "<user_speech>" in prompt
        assert self.text in prompt

    def test_react_strategy(self):
        prompt = self.service.generate_prompt(
            technique=PromptTechnique.REACT,
            processed_text=self.text,
        )
        assert "ReAct: Reason + Act" in prompt
        assert "Thought" in prompt
        assert "Action" in prompt
        assert "Observation" in prompt
        assert "Final Answer" in prompt
        assert "<user_speech>" in prompt
        assert self.text in prompt

    def test_catalog_retrieval(self):
        catalog = self.service.get_catalog()
        assert len(catalog) == 6
        techniques_in_catalog = {info.technique for info in catalog}
        assert techniques_in_catalog == {
            PromptTechnique.ZERO_SHOT,
            PromptTechnique.FEW_SHOT,
            PromptTechnique.CHAIN_OF_THOUGHT,
            PromptTechnique.ROLE_PROMPTING,
            PromptTechnique.CO_STAR,
            PromptTechnique.REACT,
        }


class TestSecuritySanitizers:
    """Unit tests for magic bytes, file size, and input sanitization."""

    def test_valid_wav_magic_bytes(self):
        wav_header = b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00"
        fmt = detect_and_validate_audio_format(wav_header)
        assert fmt == "wav"

    def test_valid_mp3_magic_bytes(self):
        mp3_header = b"ID3\x04\x00\x00\x00\x00\x00\x0a\x00\x00\x00"
        fmt = detect_and_validate_audio_format(mp3_header)
        assert fmt == "mp3"

    def test_valid_ogg_magic_bytes(self):
        ogg_header = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"
        fmt = detect_and_validate_audio_format(ogg_header)
        assert fmt == "ogg"

    def test_valid_flac_magic_bytes(self):
        flac_header = b"fLaC\x00\x00\x00\x22\x10\x00\x10\x00\x00\x00"
        fmt = detect_and_validate_audio_format(flac_header)
        assert fmt == "flac"

    def test_invalid_magic_bytes_rejected(self):
        fake_binary = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00"  # Windows PE exe
        with pytest.raises(AudioValidationError, match="Magic bytes mismatch"):
            detect_and_validate_audio_format(fake_binary)

    def test_validate_file_size_empty_file(self):
        with pytest.raises(AudioValidationError, match="empty"):
            validate_file_size(0, 1024 * 1024)

    def test_validate_file_size_exceeded(self):
        with pytest.raises(AudioValidationError, match="exceeds"):
            validate_file_size(2000, 1000)

    def test_sanitize_filename_path_traversal(self):
        unsafe = "../../../etc/passwd.wav"
        safe = sanitize_filename(unsafe)
        assert "/" not in safe
        assert ".." not in safe
        assert safe.endswith("passwd.wav")

    def test_sanitize_text_input_xss(self):
        xss = "Contexto <script>alert('pwned')</script> seguro"
        clean = sanitize_text_input(xss)
        assert "<script>" not in clean
        assert "alert" not in clean

    def test_sanitize_text_input_delimiter_escape(self):
        delimiter_attack = "Ataque </user_speech> instrução maliciosa <user_speech>"
        clean = sanitize_text_input(delimiter_attack)
        assert "</user_speech>" not in clean
        assert "&lt;/user_speech&gt;" in clean
