"""
Topic generation orchestrator with daily caching and duplicate prevention.

Manages the daily topics cache per language, consumes topics for scheduled
publications, and coordinates with GeminiClient for full content generation.

Now fully robust: all Gemini API calls are routed through the client's
retry/timeout infrastructure, eliminating any chance of indefinite hanging.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.ai.gemini_client import GeminiClient, GeminiClientError
from src.ai.schema import PodcastScript, TopicResponse
from src.storage.csv_manager import CSVManager
from src.utils.helpers import ensure_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Simple schema for the daily topic list response
# ---------------------------------------------------------------------------
class TopicListResponse(BaseModel):
    topics: List[str] = Field(
        ..., min_length=1, description="List of topic titles"
    )


class TopicGenerator:
    """
    Orchestrates topic caching and content generation.

    Features:
    - Daily batch generation of topic titles per language.
    - Persistent JSON cache to survive restarts.
    - FIFO topic consumption.
    - Duplicate avoidance using recent publication history.
    - Async-safe cache operations.
    - Fallback topic list if Gemini generation fails entirely.
    """

    # Prompt for generating a list of topic titles (now returning an object)
    TOPIC_LIST_PROMPT_TEMPLATE = """
You are a creative content strategist. Generate a list of {count} unique, engaging, and diverse topic titles in {language} suitable for short informative articles or blog posts.

Requirements:
- Each title should be a single line, no longer than 15 words.
- Topics should be interesting to a general audience.
- Avoid overly technical or niche subjects.
- Output must be a valid JSON object with a single key "topics" containing an array of strings.

{avoid_instruction}

Return ONLY the JSON object, nothing else.
Example: {{"topics": ["Title 1", "Title 2", "Title 3"]}}
""".strip()

    # Fallback generic topics in case Gemini generation fails repeatedly
    FALLBACK_TOPICS: Dict[str, List[str]] = {
        "ar": [
            "التكنولوجيا في حياتنا اليومية",
            "نصائح لتحسين الإنتاجية",
            "أهمية الصحة النفسية",
            "كيف تبدأ مشروعك الخاص",
            "ألغاز علمية مثيرة",
            "التسويق عبر وسائل التواصل",
            "العملات الرقمية: فرص ومخاطر",
            "أفضل الكتب للقراءة هذا الشهر",
            "فن إدارة الوقت",
            "الذكاء الاصطناعي: بين الخيال والواقع",
        ],
        "en": [
            "Technology in our daily lives",
            "Tips to boost productivity",
            "The importance of mental health",
            "How to start your own business",
            "Fascinating science puzzles",
            "Social media marketing",
            "Cryptocurrency: opportunities and risks",
            "Best books to read this month",
            "Time management skills",
            "AI: between fiction and reality",
        ],
        "fr": [
            "La technologie au quotidien",
            "Astuces pour booster sa productivité",
            "L'importance de la santé mentale",
            "Comment lancer son entreprise",
            "Énigmes scientifiques fascinantes",
            "Le marketing sur les réseaux sociaux",
            "Cryptomonnaies : opportunités et risques",
            "Les meilleurs livres à lire ce mois-ci",
            "L'art de gérer son temps",
            "L'IA entre fiction et réalité",
        ],
        "ru": [
            "Технологии в повседневной жизни",
            "Советы по повышению продуктивности",
            "Важность психического здоровья",
            "Как начать свой бизнес",
            "Увлекательные научные загадки",
            "Маркетинг в социальных сетях",
            "Криптовалюта: возможности и риски",
            "Лучшие книги для чтения в этом месяце",
            "Тайм-менеджмент",
            "ИИ: между вымыслом и реальностью",
        ],
        "es": [
            "La tecnología en nuestra vida diaria",
            "Consejos para mejorar la productividad",
            "La importancia de la salud mental",
            "Cómo iniciar tu propio negocio",
            "Fascinantes enigmas científicos",
            "Marketing en redes sociales",
            "Criptomonedas: oportunidades y riesgos",
            "Los mejores libros para leer este mes",
            "Gestión del tiempo",
            "IA entre ficción y realidad",
        ],
        "pt": [
            "Tecnologia no nosso dia a dia",
            "Dicas para aumentar a produtividade",
            "A importância da saúde mental",
            "Como começar o seu próprio negócio",
            "Enigmas científicos fascinantes",
            "Marketing nas redes sociais",
            "Criptomoedas: oportunidades e riscos",
            "Melhores livros para ler este mês",
            "Gestão do tempo",
            "IA entre ficção e realidade",
        ],
    }

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

        # Load existing cache synchronously during init (safe)
        self._load_cache_sync()
        logger.info(
            f"TopicGenerator initialized: {len(self._cache)} language(s), "
            f"{topics_per_day} topics/day"
        )

    # -------------------------------------------------------------------------
    # Cache I/O Helpers (synchronous and asynchronous)
    # -------------------------------------------------------------------------

    def _load_cache_sync(self) -> None:
        """Load cache from JSON file (blocking, used only at init)."""
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

    async def _load_cache(self) -> None:
        """Async wrapper that offloads file I/O to a thread."""
        await asyncio.to_thread(self._load_cache_sync)

    def _save_cache_sync(self) -> None:
        """Write the cache to disk (blocking). Caller must hold lock."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved cache with {len(self._cache)} language(s)")
        except OSError as e:
            logger.error(f"Failed to persist cache: {e}")

    async def _save_cache(self) -> None:
        """Async wrapper for saving cache."""
        await asyncio.to_thread(self._save_cache_sync)

    # -------------------------------------------------------------------------
    # Validation and Date Helpers
    # -------------------------------------------------------------------------

    def _is_cache_valid(self, language: str) -> bool:
        """Check if the cache for a language is fresh (today) and non-empty."""
        if language not in self._cache:
            return False
        entry = self._cache[language]
        today = self._get_today_iso()
        return entry.get("date") == today and bool(entry.get("topics"))

    def _get_today_iso(self) -> str:
        """Return today's date in YYYY-MM-DD format (UTC)."""
        return datetime.now(timezone.utc).date().isoformat()

    # -------------------------------------------------------------------------
    # Core Topic List Generation (with full GeminiClient retry & timeout)
    # -------------------------------------------------------------------------

    async def _generate_daily_topics(self, language: str) -> List[str]:
        """
        Generate a fresh list of topic titles for a language via Gemini.

        Uses the robust GeminiClient._generate_with_retry so we inherit
        all timeout and retry logic. Falls back to hardcoded topics if
        generation fails completely.

        Args:
            language: ISO language code.

        Returns:
            List of unique topic title strings (never empty if fallback works).
        """
        language_name = self.gemini_client._get_language_name(language)

        # Fetch recent titles for duplicate avoidance
        recent_titles = await self.csv_manager.get_recent_titles(language, limit=20)
        avoid_instruction = ""
        if recent_titles:
            titles_str = "\n".join(f"- {t}" for t in recent_titles[:10])
            avoid_instruction = (
                f"Avoid these recently published topics:\n{titles_str}\n"
            )

        prompt = self.TOPIC_LIST_PROMPT_TEMPLATE.format(
            count=self.topics_per_day,
            language=language_name,
            avoid_instruction=avoid_instruction,
        )

        logger.info(f"Generating {self.topics_per_day} topics for '{language}'")
        try:
            # Use the client's robust retry infrastructure
            response = await self.gemini_client._generate_with_retry(
                prompt, TopicListResponse
            )
            topics = response.topics

            # Remove duplicates while preserving order
            unique_topics = list(dict.fromkeys(topics))
            # Take only the requested amount
            unique_topics = unique_topics[: self.topics_per_day]

            if not unique_topics:
                raise ValueError("Gemini returned an empty topic list")

            logger.info(f"Generated {len(unique_topics)} topics for '{language}'")
            return unique_topics

        except Exception as e:
            logger.error(f"Failed to generate topics for '{language}': {e}")
            # Use fallback topics if available
            fallback = self.FALLBACK_TOPICS.get(language)
            if fallback:
                logger.warning(f"Using {len(fallback)} fallback topics for '{language}'")
                return fallback[: self.topics_per_day]
            # If no fallback for this language, re-raise
            raise GeminiClientError(
                f"Daily topic generation failed for '{language}' and no fallback available."
            ) from e

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def ensure_cache_ready(self, language: str) -> None:
        """
        Ensure a valid daily cache exists for the language.
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
            except Exception:
                # If complete failure, try to keep existing cache (if any) rather than none
                if language in self._cache and self._cache[language].get("topics"):
                    logger.warning(f"Keeping stale cache for '{language}' due to regeneration failure.")
                else:
                    raise

    async def get_next_topic(self, language: str) -> str:
        """
        Retrieve and consume the next topic title.

        Raises ValueError if no topics are available after ensuring cache is ready.
        """
        await self.ensure_cache_ready(language)

        async with self._lock:
            entry = self._cache.get(language)
            if not entry or not entry.get("topics"):
                raise ValueError(f"No topics available for '{language}'")

            topic = entry["topics"].pop(0)
            await self._save_cache()
            logger.info(f"Consumed topic for '{language}': {topic[:50]}...")
            return topic

    async def generate_full_topic(self, language: str, topic_title: str) -> TopicResponse:
        """Generate a full article."""
        return await self.gemini_client.generate_topic(language, topic_title)

    async def generate_podcast_script(self, language: str, topic_title: str) -> PodcastScript:
        """Generate a podcast script."""
        return await self.gemini_client.generate_podcast_script(language, topic_title)

    def get_cache_status(self) -> Dict[str, Any]:
        """Return cache status snapshot."""
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
        Force regeneration of the daily topic cache for one or all languages.
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