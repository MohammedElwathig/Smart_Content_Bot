"""
Topic generation orchestrator with daily caching and duplicate prevention.

Manages the daily topics cache per language, consumes topics for scheduled
publications, and coordinates with GeminiClient for full content generation.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.ai.gemini_client import GeminiClient, GeminiClientError
from src.ai.schema import PodcastScript, TopicResponse
from src.storage.csv_manager import CSVManager
from src.utils.helpers import ensure_directory, utc_now_iso
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TopicGenerator:
    """
    Orchestrates topic caching and content generation.

    Features:
    - Daily batch generation of topic titles per language.
    - Persistent JSON cache to survive restarts.
    - FIFO topic consumption.
    - Duplicate avoidance using recent publication history.
    - Async-safe cache operations.
    """

    # Prompt for generating a list of topic titles
    TOPIC_LIST_PROMPT_TEMPLATE = """
You are a creative content strategist. Generate a list of {count} unique, engaging, and diverse topic titles in {language} suitable for short informative articles or blog posts.

Requirements:
- Each title should be a single line, no longer than 15 words.
- Topics should be interesting to a general audience.
- Avoid overly technical or niche subjects.
- Output must be a valid JSON array of strings.

{avoid_instruction}

Return ONLY the JSON array, nothing else.
Example: ["Title 1", "Title 2", "Title 3"]
""".strip()

    def __init__(
        self,
        gemini_client: GeminiClient,
        csv_manager: CSVManager,
        cache_file: str = "data/daily_topics_cache.json",
        topics_per_day: int = 10,
    ) -> None:
        """
        Initialize the topic generator.

        Args:
            gemini_client: GeminiClient instance for AI generation.
            csv_manager: CSVManager instance for publication history.
            cache_file: Path to JSON cache file.
            topics_per_day: Number of topics to generate per language daily.
        """
        self.gemini_client = gemini_client
        self.csv_manager = csv_manager
        self.cache_file = cache_file
        self.topics_per_day = topics_per_day

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

        # Ensure cache directory exists
        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir:
            ensure_directory(cache_dir)

        self._load_cache()
        logger.info(
            f"TopicGenerator initialized: {len(self._cache)} language(s), "
            f"{topics_per_day} topics/day"
        )

    # -------------------------------------------------------------------------
    # Private Cache Helpers
    # -------------------------------------------------------------------------

    def _load_cache(self) -> None:
        """Load cache from JSON file; create empty on failure."""
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            logger.debug(f"Loaded cache with {len(self._cache)} language(s)")
        except FileNotFoundError:
            logger.info("No existing cache file, starting fresh")
            self._cache = {}
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted cache file: {e}, starting fresh")
            self._cache = {}

    async def _save_cache(self) -> None:
        """Persist cache to JSON file (caller must hold lock)."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved cache with {len(self._cache)} language(s)")
        except OSError as e:
            logger.error(f"Failed to save cache: {e}")

    def _is_cache_valid(self, language: str) -> bool:
        """
        Check if cache for a language is valid (today's date, non-empty topics).
        """
        if language not in self._cache:
            return False
        entry = self._cache[language]
        today = datetime.utcnow().date().isoformat()
        return entry.get("date") == today and bool(entry.get("topics"))

    def _get_today_iso(self) -> str:
        """Return today's date in YYYY-MM-DD format (UTC)."""
        return datetime.utcnow().date().isoformat()

    # -------------------------------------------------------------------------
    # Daily Topic List Generation
    # -------------------------------------------------------------------------

    async def _generate_daily_topics(self, language: str) -> List[str]:
        """
        Call Gemini to generate a fresh list of topic titles for a language.

        Args:
            language: ISO language code.

        Returns:
            List of topic title strings.

        Raises:
            GeminiClientError: If generation fails after retries.
        """
        # Get language name for prompt
        language_name = {"ar": "Arabic", "en": "English", "fr": "French"}.get(
            language, language.upper()
        )

        # Fetch recent titles for duplicate avoidance
        recent_titles = await self.csv_manager.get_recent_titles(
            language, limit=20
        )
        avoid_instruction = ""
        if recent_titles:
            titles_str = "\n".join(f"- {t}" for t in recent_titles[:10])
            avoid_instruction = (
                f"Avoid these recently published topics:\n{titles_str}\n"
            )
        else:
            avoid_instruction = ""

        prompt = self.TOPIC_LIST_PROMPT_TEMPLATE.format(
            count=self.topics_per_day,
            language=language_name,
            avoid_instruction=avoid_instruction,
        )

        logger.info(f"Generating {self.topics_per_day} topics for '{language}'")
        try:
            # Use a simpler call without full schema; we just need a list
            import google.generativeai as genai
            from google.generativeai.types import GenerationConfig

            api_key = self.gemini_client.key_manager.get_next_key()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.gemini_client.model_name)

            response = await model.generate_content_async(
                prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.9,  # More creative for topics
                ),
            )
            # Parse JSON list
            topics = json.loads(response.text)
            if isinstance(topics, list) and all(isinstance(t, str) for t in topics):
                # Ensure uniqueness
                topics = list(dict.fromkeys(topics))
                logger.info(f"Generated {len(topics)} topics for '{language}'")
                return topics[: self.topics_per_day]
            else:
                logger.error(f"Invalid topic list format: {response.text[:200]}")
                raise GeminiClientError("Topic list response not a JSON array of strings")

        except Exception as e:
            logger.error(f"Failed to generate daily topics for '{language}': {e}")
            raise GeminiClientError(f"Daily topic generation failed: {e}") from e

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    async def ensure_cache_ready(self, language: str) -> None:
        """
        Ensure a valid daily cache exists for the given language.
        Regenerates if missing or stale.
        """
        async with self._lock:
            if self._is_cache_valid(language):
                return

            logger.info(f"Cache for '{language}' is stale/missing. Regenerating...")
            try:
                topics = await self._generate_daily_topics(language)
                self._cache[language] = {
                    "date": self._get_today_iso(),
                    "topics": topics,
                }
                await self._save_cache()
            except Exception as e:
                logger.error(f"Cache regeneration failed for '{language}': {e}")
                raise

    async def get_next_topic(self, language: str) -> str:
        """
        Retrieve and consume the next topic title for the language.

        Args:
            language: ISO language code.

        Returns:
            Topic title string.

        Raises:
            ValueError: If cache is empty after ensuring readiness.
        """
        await self.ensure_cache_ready(language)

        async with self._lock:
            entry = self._cache.get(language)
            if not entry or not entry.get("topics"):
                # This should not happen after ensure_cache_ready
                raise ValueError(f"No topics available for '{language}'")

            topic = entry["topics"].pop(0)
            await self._save_cache()
            logger.info(f"Consumed topic for '{language}': {topic[:50]}...")
            return topic

    async def generate_full_topic(
        self, language: str, topic_title: str
    ) -> TopicResponse:
        """
        Generate a complete article and quote for the given topic.

        Args:
            language: ISO language code.
            topic_title: The specific topic title.

        Returns:
            Validated TopicResponse object.
        """
        logger.info(f"Generating full topic for '{language}': {topic_title[:50]}...")
        return await self.gemini_client.generate_topic(language, topic_title)

    async def generate_podcast_script(
        self, language: str, topic_title: str
    ) -> PodcastScript:
        """
        Generate a podcast script for the given topic.

        Args:
            language: ISO language code.
            topic_title: The topic title.

        Returns:
            Validated PodcastScript object.
        """
        logger.info(f"Generating podcast script for '{language}': {topic_title[:50]}...")
        return await self.gemini_client.generate_podcast_script(language, topic_title)

    def get_cache_status(self) -> Dict[str, Any]:
        """
        Return a summary of the current cache status.

        Returns:
            Dictionary with per-language date and remaining topics count.
        """
        status = {}
        for lang, entry in self._cache.items():
            status[lang] = {
                "date": entry.get("date", "unknown"),
                "remaining": len(entry.get("topics", [])),
                "is_valid": self._is_cache_valid(lang),
            }
        return status

    async def regenerate_cache(self, language: Optional[str] = None) -> None:
        """
        Force regeneration of the daily topic cache.

        Args:
            language: Specific language to regenerate. If None, regenerates all.
        """
        from src.storage.languages import get_language_codes

        if language is not None:
            languages = [language]
        else:
            languages = get_language_codes()

        async with self._lock:
            for lang in languages:
                logger.info(f"Force regenerating cache for '{lang}'")
                try:
                    topics = await self._generate_daily_topics(lang)
                    self._cache[lang] = {
                        "date": self._get_today_iso(),
                        "topics": topics,
                    }
                except Exception as e:
                    logger.error(f"Failed to regenerate cache for '{lang}': {e}")
            await self._save_cache()