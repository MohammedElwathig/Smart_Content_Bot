"""
Asynchronous Gemini API client with intelligent key rotation and schema validation.

Handles all communication with Google Gemini, including prompt construction,
retries on quota exhaustion, and parsing responses into Pydantic models.
"""

import asyncio
from typing import Dict, Optional, Type, TypeVar

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from pydantic import BaseModel, ValidationError

from src.ai.key_manager import AllKeysExhaustedError, GeminiKeyManager
from src.ai.schema import PodcastScript, TopicResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Type variable for generic schema parsing
T = TypeVar("T", bound=BaseModel)


class GeminiClientError(Exception):
    """Raised when a non-retryable Gemini API error occurs."""

    pass


class GeminiClient:
    """
    Async client for Google Gemini with key rotation and structured output.
    """

    # Language name mapping for better prompt context
    LANGUAGE_NAMES: Dict[str, str] = {
        "ar": "Arabic",
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "de": "German",
        "zh": "Chinese",
        "ja": "Japanese",
    }

    # Default prompt templates
    TOPIC_PROMPT_TEMPLATE = """
You are an expert content creator and writer. Write a comprehensive article in {language} on the topic: "{topic_title}".

The response must be a valid JSON object with the following structure:
{{
    "title": "A compelling title in {language}",
    "introduction": "An engaging opening paragraph of 2-4 sentences",
    "body": [
        "First main paragraph with detailed information",
        "Second main paragraph with supporting details",
        "Third main paragraph with examples or context",
        "Fourth main paragraph (optional)"
    ],
    "conclusion": "A closing paragraph that summarizes the key message",
    "quote_text": "A relevant, inspiring quote in {language} related to the topic",
    "quote_author": "The author of the quote (use 'Unknown' if not attributable)"
}}

Important guidelines:
- Write naturally and fluently in {language}.
- The body must contain between 3 and 5 paragraphs, each with 2-4 sentences.
- The quote should be authentic and thought-provoking.
- Do not include any text outside the JSON object.
- Ensure the JSON is valid and uses double quotes.
""".strip()

    PODCAST_PROMPT_TEMPLATE = """
You are a podcast scriptwriter. Write a short, engaging podcast script in {language} based on the topic: "{topic_title}".

The response must be a valid JSON object with exactly this structure:
{{
    "title": "A catchy episode title in {language}",
    "intro": "A warm, brief introduction (20-40 words) that hooks the listener",
    "segments": [
        "First key point or insight (1-2 sentences)",
        "Second key point or insight (1-2 sentences)",
        "Third key point or insight (1-2 sentences)"
    ],
    "outro": "A brief conclusion (20-40 words) with a call to action or final thought"
}}

Important guidelines:
- Write conversationally in {language}, as if speaking to a friend.
- The segments list must contain between 2 and 4 items.
- Keep the total script under 500 words for a short 1-2 minute podcast.
- Do not include any text outside the JSON object.
- Ensure the JSON is valid and uses double quotes.
""".strip()

    def __init__(
        self,
        key_manager: GeminiKeyManager,
        model_name: str = "gemini-2.0-flash-exp",
    ) -> None:
        """
        Initialize the Gemini client.

        Args:
            key_manager: Instance of GeminiKeyManager for key rotation.
            model_name: Gemini model identifier (e.g., 'gemini-2.0-flash-exp').
        """
        self.key_manager = key_manager
        self.model_name = model_name
        logger.info(f"GeminiClient initialized with model: {model_name}")

    def _get_language_name(self, language_code: str) -> str:
        """Return human-readable language name for prompt context."""
        return self.LANGUAGE_NAMES.get(language_code, language_code.upper())

    def _build_topic_prompt(self, language: str, topic_title: str) -> str:
        """Build the prompt for full topic generation."""
        language_name = self._get_language_name(language)
        return self.TOPIC_PROMPT_TEMPLATE.format(
            language=language_name, topic_title=topic_title
        )

    def _build_podcast_prompt(self, language: str, topic_title: str) -> str:
        """Build the prompt for podcast script generation."""
        language_name = self._get_language_name(language)
        return self.PODCAST_PROMPT_TEMPLATE.format(
            language=language_name, topic_title=topic_title
        )

    async def _generate_with_retry(
        self,
        prompt: str,
        schema_class: Type[T],
        max_retries: Optional[int] = None,
    ) -> T:
        """
        Internal method to handle API call with key rotation and retries.

        Args:
            prompt: The full prompt to send to Gemini.
            schema_class: Pydantic model class for validation.
            max_retries: Maximum number of retry attempts. If None, uses total keys count.

        Returns:
            Validated Pydantic model instance.

        Raises:
            AllKeysExhaustedError: If no valid keys remain.
            GeminiClientError: For non-retryable API errors after all retries.
        """
        # Determine maximum retries
        if max_retries is None:
            # Get total keys count from status (avoid accessing private member)
            status = await self.key_manager.get_status()
            max_retries = status["total_keys"]

        last_error = None
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                # Get next available key (async)
                api_key = await self.key_manager.get_next_key()
            except AllKeysExhaustedError:
                logger.error("No available Gemini API keys")
                raise

            redacted = f"{api_key[:8]}...{api_key[-4:]}"
            logger.debug(f"Attempt {attempt + 1}/{max_retries} with key {redacted}")

            try:
                # Configure the client with current key
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(self.model_name)

                # Create generation config with structured output
                generation_config = GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema_class,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=2048,
                )

                # Make async API call
                response = await model.generate_content_async(
                    prompt, generation_config=generation_config
                )

                # Validate and parse response
                try:
                    parsed = schema_class.model_validate_json(response.text)
                    logger.info(
                        f"Gemini generation succeeded with key {redacted} "
                        f"({len(response.text)} chars)"
                    )
                    return parsed
                except ValidationError as e:
                    logger.error(f"Schema validation failed: {e}")
                    # Retry; model may produce invalid JSON occasionally
                    last_error = e
                    continue

            except Exception as e:
                error_str = str(e)
                # Check for quota exhaustion (HTTP 429)
                if "429" in error_str or "Resource exhausted" in error_str:
                    logger.warning(f"Key {redacted} exhausted: {error_str}")
                    await self.key_manager.mark_key_exhausted(api_key)
                    last_error = e
                    # Brief pause before next key
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 2.0)  # Cap at 2 seconds
                    continue
                elif "400" in error_str or "Invalid" in error_str:
                    # Bad request - likely prompt issue, don't retry
                    logger.error(f"Bad request with key {redacted}: {error_str}")
                    raise GeminiClientError(f"Gemini API bad request: {error_str}")
                else:
                    # Other errors (500, network) - retry with same key after delay
                    logger.warning(f"API error (retryable): {error_str}")
                    last_error = e
                    await asyncio.sleep(2.0 ** attempt)  # Exponential backoff
                    continue

        # If we exhaust retries
        if last_error:
            raise GeminiClientError(
                f"Gemini generation failed after {max_retries} attempts"
            ) from last_error
        raise GeminiClientError("Unknown error during Gemini generation")

    async def generate_topic(self, language: str, topic_title: str) -> TopicResponse:
        """
        Generate a full article and quote for the given topic.

        Args:
            language: ISO language code (e.g., 'ar', 'en').
            topic_title: The specific topic to write about.

        Returns:
            Validated TopicResponse object.

        Raises:
            AllKeysExhaustedError: If no API keys are available.
            GeminiClientError: For non-retryable failures.
        """
        prompt = self._build_topic_prompt(language, topic_title)
        logger.info(f"Generating topic for '{language}': {topic_title[:50]}...")
        return await self._generate_with_retry(prompt, TopicResponse)

    async def generate_podcast_script(
        self, language: str, topic_title: str
    ) -> PodcastScript:
        """
        Generate a short podcast script for the given topic.

        Args:
            language: ISO language code.
            topic_title: The topic for the podcast.

        Returns:
            Validated PodcastScript object.

        Raises:
            AllKeysExhaustedError: If no API keys are available.
            GeminiClientError: For non-retryable failures.
        """
        prompt = self._build_podcast_prompt(language, topic_title)
        logger.info(f"Generating podcast script for '{language}': {topic_title[:50]}...")
        return await self._generate_with_retry(prompt, PodcastScript)

    async def test_connection(self, api_key: Optional[str] = None) -> bool:
        """
        Test if a Gemini API key is valid by making a minimal call.

        Args:
            api_key: Specific key to test. If None, uses get_next_key().

        Returns:
            True if connection succeeds, False otherwise.
        """
        try:
            if api_key is None:
                api_key = await self.key_manager.get_next_key()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model_name)
            response = await model.generate_content_async("Say 'OK'")
            return bool(response.text)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False