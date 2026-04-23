"""
Edge-TTS integration for podcast audio generation.

Provides a robust async interface to Microsoft Edge's free Text-to-Speech service.
Features:
- Language-specific neural voice selection.
- Configurable speaking rate and pitch via environment variables.
- Automatic retry on transient network failures.
- Seamless integration with PodcastScript Pydantic models.
"""

import asyncio
import os
from typing import Dict, Optional

import edge_tts
from edge_tts import exceptions as edge_exceptions

from config.settings import settings
from src.ai.schema import PodcastScript
from src.utils.helpers import ensure_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Default Voice Mapping
# These are high-quality neural voices available in Edge-TTS.
# For a complete list, refer to:
# https://github.com/microsoft/cognitive-services-speech-sdk-js/blob/master/src/common.browser/VoiceList.ts
# -----------------------------------------------------------------------------
DEFAULT_VOICE_MAP: Dict[str, str] = {
    "ar": "ar-EG-SalmaNeural",      # Arabic (Egypt)
    "en": "en-US-JennyNeural",      # English (US)
    "fr": "fr-FR-DeniseNeural",     # French (France)
    "es": "es-ES-ElviraNeural",     # Spanish (Spain)
    "de": "de-DE-KatjaNeural",      # German
    "it": "it-IT-ElsaNeural",       # Italian
    "pt": "pt-BR-FranciscaNeural",  # Portuguese (Brazil)
    "ru": "ru-RU-SvetlanaNeural",   # Russian
    "zh": "zh-CN-XiaoxiaoNeural",   # Chinese (Mandarin)
    "ja": "ja-JP-NanamiNeural",     # Japanese
    "ko": "ko-KR-SunHiNeural",      # Korean
}

# Fallback voice when a language is not explicitly mapped
FALLBACK_VOICE: str = "en-US-JennyNeural"

# Number of retry attempts for transient network issues
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0


class TTSError(Exception):
    """Custom exception raised when audio generation fails irrecoverably."""
    pass


class EdgeTTSService:
    """
    Asynchronous service for generating MP3 audio from text using Microsoft Edge TTS.

    This service is completely free and does not require an API key.
    It uses the same voices available in Microsoft Edge's "Read Aloud" feature.

    Usage:
        tts = EdgeTTSService()
        await tts.generate_audio("Hello world", "en", "/tmp/audio.mp3")
    """

    def __init__(self) -> None:
        """
        Initialize the TTS service.

        Configuration is loaded from the global `settings` object, which reads
        environment variables:
        - TTS_RATE: Speaking rate (e.g., "+0%", "-10%")
        - TTS_PITCH: Pitch adjustment (e.g., "+0Hz", "-2Hz")
        - TTS_VOICE_<lang>: Override default voice per language.
        """
        self.rate = settings.tts_rate
        self.pitch = settings.tts_pitch

        # Start with the built-in default voices
        self.voice_map = DEFAULT_VOICE_MAP.copy()
        self._apply_voice_overrides()

        logger.info(
            f"EdgeTTSService initialized: rate='{self.rate}', pitch='{self.pitch}', "
            f"{len(self.voice_map)} language(s) configured."
        )

    def _apply_voice_overrides(self) -> None:
        """
        Merge any voice overrides defined in environment variables.

        Environment variables take precedence over the defaults.
        Example: TTS_VOICE_ar=ar-SA-ZariyaNeural
        """
        overrides = settings.tts_voice_overrides
        if overrides:
            for lang, voice in overrides.items():
                self.voice_map[lang] = voice
                logger.debug(f"Voice override applied: '{lang}' -> '{voice}'")
        else:
            logger.debug("No voice overrides found in environment.")

    def _get_voice_for_language(self, language: str) -> str:
        """
        Resolve the Edge-TTS voice identifier for a given ISO language code.

        Args:
            language: ISO language code (e.g., 'ar', 'en').

        Returns:
            A valid Edge-TTS voice short name (e.g., 'ar-EG-SalmaNeural').
        """
        voice = self.voice_map.get(language)
        if voice is None:
            logger.warning(
                f"No voice configured for language code '{language}'. "
                f"Falling back to '{FALLBACK_VOICE}'."
            )
            voice = FALLBACK_VOICE
        return voice

    async def generate_audio(
        self,
        text: str,
        language: str,
        output_path: str,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        """
        Convert plain text to speech and save as an MP3 file.

        Implements automatic retry on transient failures (network timeouts,
        temporary service unavailability). Irrecoverable errors (e.g., invalid
        voice) raise TTSError immediately.

        Args:
            text: Plain text to synthesize.
            language: Language code for voice selection.
            output_path: Absolute or relative path where the MP3 file will be saved.
            max_retries: Number of retry attempts for transient errors.

        Raises:
            TTSError: If audio generation fails after all retries, or if a
                      non-retryable error occurs.
            ValueError: If the text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text for TTS cannot be empty.")

        # Ensure the destination directory exists.
        directory = os.path.dirname(output_path)
        if directory:
            await asyncio.to_thread(ensure_directory, directory)

        voice = self._get_voice_for_language(language)

        logger.info(
            f"Starting TTS: language='{language}', voice='{voice}', "
            f"text_length={len(text)} chars, output='{output_path}'"
        )

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # Create a Communicate instance with the given parameters.
                # Note: edge_tts.Communicate does not accept 'options' string;
                # rate and pitch are passed directly.
                communicate = edge_tts.Communicate(
                    text=text, voice=voice, rate=self.rate, pitch=self.pitch
                )

                # Perform the async save operation.
                await communicate.save(output_path)

                # Verify the file was actually created and has size > 0.
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    size_kb = os.path.getsize(output_path) / 1024
                    logger.info(f"Audio successfully saved: '{output_path}' ({size_kb:.1f} KB)")
                    return
                else:
                    logger.warning("Edge-TTS reported success but output file is missing or empty.")
                    raise TTSError("Audio file was not created or is empty.")

            except edge_exceptions.NoAudioReceived as e:
                # This usually indicates the text was invalid or contained only
                # unspeakable characters (e.g., only punctuation). Not retryable.
                logger.error(f"No audio received from Edge-TTS: {e}")
                raise TTSError("Edge-TTS returned no audio (check input text).") from e

            except (edge_exceptions.UnknownResponse, edge_exceptions.UnexpectedResponse) as e:
                # These are likely transient server-side issues. Retry.
                logger.warning(f"Transient Edge-TTS error (attempt {attempt}/{max_retries}): {e}")
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)

            except Exception as e:
                # Catch-all for any other unexpected errors (network, filesystem).
                logger.error(f"Unexpected error during TTS generation: {type(e).__name__}: {e}")
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)

        # If we exit the loop, all retries failed.
        logger.error(f"TTS generation failed after {max_retries} attempts.")
        raise TTSError(f"Audio generation failed after {max_retries} retries.") from last_error

    async def generate_audio_from_script(
        self,
        script: PodcastScript,
        language: str,
        output_path: str,
    ) -> None:
        """
        Generate an MP3 audio file directly from a PodcastScript object.

        The script's intro, segments, and outro are concatenated with natural pauses.

        Args:
            script: Validated PodcastScript Pydantic model.
            language: Language code for voice selection.
            output_path: Destination path for the MP3 file.
        """
        # Build the full spoken text.
        # Use double newline between sections for a brief pause.
        parts = [
            script.intro.strip(),
            " ".join(script.segments).strip(),
            script.outro.strip(),
        ]
        # Filter out any empty parts
        parts = [p for p in parts if p]

        # Join with a delimiter that Edge-TTS interprets as a pause.
        full_text = "\n\n".join(parts)

        logger.debug(
            f"Constructed TTS text from script: {len(full_text)} chars, "
            f"word count ~{script.estimated_word_count()}"
        )

        await self.generate_audio(full_text, language, output_path)