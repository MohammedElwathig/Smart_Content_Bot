"""
Asynchronous Gemini API client with intelligent key rotation and schema validation.

Handles all communication with Google Gemini, including prompt construction,
retries on quota exhaustion, and parsing responses into Pydantic models.

Enhanced with:
- Mandatory timeouts on all API calls to prevent indefinite hanging.
- Exponential backoff with random jitter for robust retry behavior.
- Granular error classification (rate limit, server error, network, validation).
- Safe logging that never leaks API keys.
"""

import asyncio
import random
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

# ---------------------------------------------------------------------------
# Constants for retry behavior
# ---------------------------------------------------------------------------
# Maximum time to wait for a single Gemini API call
API_TIMEOUT_SECONDS = 90.0
# Initial delay between retries (seconds)
INITIAL_RETRY_DELAY = 1.0
# Maximum delay between retries (seconds)
MAX_RETRY_DELAY = 30.0
# Backoff multiplier
BACKOFF_FACTOR = 2.0
# Random jitter factor (0.0 to 1.0) to avoid thundering herd
JITTER_FACTOR = 0.1


class GeminiClientError(Exception):
    """Raised when a non-retryable Gemini API error occurs or all retries are exhausted."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_error = original_error


class GeminiClient:
    """
    Async client for Google Gemini with key rotation and structured output.

    Features:
    - Automatic key rotation on 429 errors.
    - Exponential backoff with jitter for transient failures.
    - Mandatory timeouts to prevent hanging.
    - Structured output parsing with Pydantic validation.
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
            model_name: Gemini model identifier.
        """
        if not isinstance(key_manager, GeminiKeyManager):
            raise TypeError("key_manager must be an instance of GeminiKeyManager")

        self.key_manager = key_manager
        self.model_name = model_name
        logger.info(f"GeminiClient initialized with model: {model_name}")

    # -----------------------------------------------------------------------
    # Language helpers
    # -----------------------------------------------------------------------

    def _get_language_name(self, language_code: str) -> str:
        """Return human-readable language name for prompt context."""
        return self.LANGUAGE_NAMES.get(language_code, language_code.upper())

    # -----------------------------------------------------------------------
    # Prompt builders
    # -----------------------------------------------------------------------

    def _build_topic_prompt(self, language: str, topic_title: str) -> str:
        """Build the prompt for full topic generation."""
        return self.TOPIC_PROMPT_TEMPLATE.format(
            language=self._get_language_name(language),
            topic_title=topic_title,
        )

    def _build_podcast_prompt(self, language: str, topic_title: str) -> str:
        """Build the prompt for podcast script generation."""
        return self.PODCAST_PROMPT_TEMPLATE.format(
            language=self._get_language_name(language),
            topic_title=topic_title,
        )

    # -----------------------------------------------------------------------
    # Core retry logic with timeout, backoff, and jitter
    # -----------------------------------------------------------------------

    @staticmethod
    def _redact_key(key: str) -> str:
        """Return a safely redacted version of an API key for logging."""
        if len(key) <= 12:
            return "***"
        return f"{key[:6]}...{key[-4:]}"

    @staticmethod
    def _classify_error(error: Exception) -> str:
        """
        Classify an exception into a retry strategy category.

        Returns one of:
            "rate_limited"  - 429 / quota exhausted
            "server_error"  - 500, 503, etc.
            "network"       - timeout, connection error
            "bad_request"   - 400, invalid prompt
            "parse_error"   - JSON decode or validation
            "unknown"       - everything else
        """
        error_str = str(error).lower()

        # Rate limiting
        if any(kw in error_str for kw in ("429", "resource exhausted", "quota")):
            return "rate_limited"

        # Server-side issues
        if any(kw in error_str for kw in ("500", "502", "503", "server error", "internal")):
            return "server_error"

        # Network issues
        if isinstance(error, (asyncio.TimeoutError, ConnectionError, TimeoutError)):
            return "network"
        if any(kw in error_str for kw in ("timeout", "timed out", "connection", "network")):
            return "network"

        # Bad request (likely our fault, don't retry blindly)
        if any(kw in error_str for kw in ("400", "invalid argument", "bad request")):
            return "bad_request"

        # Parsing / validation
        if isinstance(error, (ValidationError, json.JSONDecodeError)):  # Note: json is imported below
            return "parse_error"

        return "unknown"

    async def _call_gemini_api(
        self, prompt: str, schema_class: Type[T], api_key: str
    ) -> T:
        """
        Execute a single API call to Gemini with a timeout.

        Args:
            prompt: The full prompt to send.
            schema_class: Pydantic model for response validation.
            api_key: The API key to use for this call.

        Returns:
            Validated Pydantic model instance.

        Raises:
            Various exceptions on failure (see _classify_error).
        """
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model_name)

        generation_config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=schema_class,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
        )

        # --- The critical timeout wrapper ---
        response = await asyncio.wait_for(
            model.generate_content_async(prompt, generation_config=generation_config),
            timeout=API_TIMEOUT_SECONDS,
        )

        # Handle empty or missing response text
        if not response or not hasattr(response, 'text') or not response.text:
            raise GeminiClientError("Gemini returned an empty or invalid response")

        # Parse and validate
        parsed = schema_class.model_validate_json(response.text)
        return parsed

    async def _generate_with_retry(
        self,
        prompt: str,
        schema_class: Type[T],
        max_retries: Optional[int] = None,
    ) -> T:
        """
        Generate content with automatic key rotation and exponential backoff.

        Args:
            prompt: The full prompt to send.
            schema_class: Pydantic model for validation.
            max_retries: Maximum retry attempts. Defaults to number of keys.

        Returns:
            Validated Pydantic model.

        Raises:
            AllKeysExhaustedError: If no keys remain.
            GeminiClientError: After all retries are exhausted.
        """
        import json  # Local import for JSON decode error

        # Determine max retries
        if max_retries is None:
            status = await self.key_manager.get_status()
            max_retries = status["total_keys"]

        last_error: Optional[Exception] = None
        current_delay = INITIAL_RETRY_DELAY

        for attempt in range(max_retries):
            # --- Get next API key ---
            try:
                api_key = await self.key_manager.get_next_key()
            except AllKeysExhaustedError:
                logger.error("All Gemini API keys exhausted. Cannot proceed.")
                raise

            redacted = self._redact_key(api_key)
            logger.debug(f"Gemini attempt {attempt + 1}/{max_retries} with key {redacted}")

            try:
                # --- Execute the API call with timeout ---
                result = await self._call_gemini_api(prompt, schema_class, api_key)
                logger.info(
                    f"Gemini generation succeeded on attempt {attempt + 1} "
                    f"(key: {redacted})"
                )
                return result

            except asyncio.TimeoutError as e:
                category = "network"
                last_error = e
                logger.warning(f"Gemini API timed out after {API_TIMEOUT_SECONDS}s (key: {redacted})")

            except ValidationError as e:
                category = "parse_error"
                last_error = e
                logger.warning(f"Gemini response failed schema validation: {e}")

            except Exception as e:
                last_error = e
                category = self._classify_error(e)
                logger.warning(
                    f"Gemini API error (category={category}, key={redacted}): {e}"
                )

            # --- Determine retry strategy based on category ---
            if category == "rate_limited":
                # Mark key as exhausted and switch to next key
                await self.key_manager.mark_key_exhausted(api_key)
                logger.warning(f"Key {redacted} marked exhausted. Rotating.")
                current_delay = INITIAL_RETRY_DELAY  # Reset delay for new key
                # Continue immediately to next key (no sleep)

            elif category in ("network", "server_error"):
                # Exponential backoff with jitter
                jitter = random.uniform(0, current_delay * JITTER_FACTOR)
                wait_time = current_delay + jitter
                logger.debug(f"Retrying in {wait_time:.1f}s (backoff + jitter)")
                await asyncio.sleep(wait_time)
                current_delay = min(current_delay * BACKOFF_FACTOR, MAX_RETRY_DELAY)

            elif category == "parse_error":
                # Invalid JSON or schema: retry with same key after short delay
                logger.debug("Retrying after parse error in 1s")
                await asyncio.sleep(1.0)

            elif category == "bad_request":
                # Likely a prompt issue. Do NOT retry.
                raise GeminiClientError(
                    f"Bad request to Gemini API: {last_error}"
                ) from last_error

            else:  # unknown
                # Conservative: wait and retry
                logger.debug(f"Unknown error, retrying in {current_delay:.1f}s")
                await asyncio.sleep(current_delay)
                current_delay = min(current_delay * BACKOFF_FACTOR, MAX_RETRY_DELAY)

        # All attempts exhausted
        raise GeminiClientError(
            f"Gemini generation failed after {max_retries} attempt(s). "
            f"Last error category: {category if 'category' in dir() else 'unknown'}"
        ) from last_error

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

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
            response = await asyncio.wait_for(
                model.generate_content_async("Say 'OK'"),
                timeout=15.0,
            )
            return bool(response.text)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False