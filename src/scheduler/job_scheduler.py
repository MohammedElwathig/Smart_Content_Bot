"""
Job scheduler for automated periodic publication.

Uses APScheduler to trigger publication jobs at configured intervals,
coordinating all services: topic generation, image creation, TTS, and Telegram.

Enhanced with per-language timeouts and safe-guards against hanging.
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import settings
from src.ai.topic_generator import TopicGenerator
from src.bot.telegram_bot import TelegramBot
from src.image.image_generator import ImageGenerator
from src.storage.csv_manager import CSVManager
from src.storage.languages import get_channel_id, get_language_codes, get_languages
from src.tts.audio_decision import should_generate_audio
from src.tts.edge_tts_service import EdgeTTSService
from src.utils.helpers import safe_delete, generate_unique_filename
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Maximum time allowed for publishing one language before timeout
PER_LANGUAGE_TIMEOUT_SECONDS = 300  # 5 minutes


class JobScheduler:
    """
    Orchestrates scheduled publication jobs using APScheduler.
    """

    def __init__(
        self,
        topic_gen: TopicGenerator,
        image_gen: ImageGenerator,
        tts_service: EdgeTTSService,
        telegram_bot: TelegramBot,
        csv_manager: CSVManager,
        interval_minutes: Optional[int] = None,
    ) -> None:
        self.topic_gen = topic_gen
        self.image_gen = image_gen
        self.tts_service = tts_service
        self.telegram_bot = telegram_bot
        self.csv_manager = csv_manager
        self.interval_minutes = interval_minutes or settings.publish_interval_minutes

        self.scheduler = AsyncIOScheduler()
        self._is_running = False

        self.last_run_time: Optional[datetime] = None
        self.last_run_status: Dict[str, Any] = {}

        logger.info(
            f"JobScheduler initialized with interval: {self.interval_minutes} minutes"
        )

    async def start(self) -> None:
        if self._is_running:
            logger.warning("Scheduler already running")
            return

        self.scheduler.add_job(
            self._publish_job,
            trigger=IntervalTrigger(minutes=self.interval_minutes),
            id="publish_job",
            name="Publication Job",
            coalesce=True,
            max_instances=1,
            misfire_grace_time=300,
        )

        self.scheduler.start()
        self._is_running = True
        logger.info(f"Scheduler started. Next publication in ~{self.interval_minutes} minutes")

    async def stop(self) -> None:
        if not self._is_running:
            return
        self.scheduler.shutdown(wait=True)
        self._is_running = False
        logger.info("Scheduler stopped")

    async def force_publish(self, language: Optional[str] = None) -> Dict[str, Any]:
        if language:
            languages_to_process = [language] if language in get_language_codes() else []
            if not languages_to_process:
                logger.error(f"Invalid language requested: {language}")
                return {"error": f"Language '{language}' not configured"}
        else:
            languages_to_process = get_language_codes()

        logger.info(f"Manual publication triggered for {len(languages_to_process)} language(s)")
        results = {}
        for lang in languages_to_process:
            try:
                success = await self._publish_for_language(lang)
                results[lang] = "success" if success else "failed"
            except asyncio.TimeoutError:
                logger.error(f"Manual publish for '{lang}' timed out after {PER_LANGUAGE_TIMEOUT_SECONDS}s")
                results[lang] = "timeout"
            except Exception as e:
                logger.exception(f"Manual publish for '{lang}' crashed")
                results[lang] = f"error: {str(e)[:100]}"
        return results

    def get_status(self) -> Dict[str, Any]:
        next_run = None
        job = self.scheduler.get_job("publish_job")
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()

        return {
            "is_running": self._is_running,
            "interval_minutes": self.interval_minutes,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "next_run_time": next_run,
            "last_run_status": self.last_run_status,
        }

    async def _publish_job(self) -> None:
        logger.info("=" * 50)
        logger.info("Starting scheduled publication cycle")
        self.last_run_time = datetime.now(timezone.utc)
        self.last_run_status = {}

        languages = get_languages()
        for lang_config in languages:
            lang = lang_config.code
            try:
                success = await self._publish_for_language(lang)
                self.last_run_status[lang] = "success" if success else "failed"
            except asyncio.TimeoutError:
                logger.error(f"Scheduled publish for '{lang}' timed out")
                self.last_run_status[lang] = "timeout"
            except Exception as e:
                logger.exception(f"Unhandled error publishing for '{lang}'")
                self.last_run_status[lang] = f"error: {str(e)[:50]}"

        logger.info("Scheduled publication cycle completed")
        logger.info("=" * 50)

    async def _publish_for_language(self, language: str) -> bool:
        """
        Execute the full publication pipeline for a single language,
        with an overall timeout to prevent indefinite hanging.
        """
        try:
            return await asyncio.wait_for(
                self._publish_for_language_impl(language),
                timeout=PER_LANGUAGE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(f"[{language}] Publication timed out after {PER_LANGUAGE_TIMEOUT_SECONDS}s")
            return False
        except Exception:
            # Already logged inside _impl, but safety net
            logger.exception(f"[{language}] Unexpected error in publish wrapper")
            return False

    async def _publish_for_language_impl(self, language: str) -> bool:
        logger.info(f"--- Publishing for language: {language} ---")

        channel_id = get_channel_id(language)
        if not channel_id:
            logger.error(f"No channel ID configured for language '{language}'")
            return False

        # Ensure channel_id is int (defensive)
        try:
            channel_id = int(channel_id)
        except (TypeError, ValueError):
            logger.error(f"Invalid channel_id for '{language}': {channel_id!r}")
            return False

        temp_files: List[str] = []
        audio_generated = "false"

        try:
            # Step 1: Get next topic title
            topic_title = await self.topic_gen.get_next_topic(language)
            logger.info(f"[{language}] Topic: {topic_title}")

            # Step 2: Generate full content
            topic = await self.topic_gen.generate_full_topic(language, topic_title)

            # Step 3: Generate article image (optional)
            article_img_path: Optional[str] = None
            try:
                article_img_path = await self.image_gen.generate_article_image(
                    topic.title, language
                )
                if article_img_path:
                    temp_files.append(article_img_path)
                    logger.info(f"[{language}] Article image generated")
            except Exception as e:
                logger.error(f"[{language}] Article image generation failed: {e}")

            # Step 3b: Generate quote image (optional)
            quote_img_path: Optional[str] = None
            try:
                quote_img_path = await self.image_gen.generate_quote_image(
                    topic.quote_text, topic.quote_author or "", language
                )
                if quote_img_path:
                    temp_files.append(quote_img_path)
                    logger.info(f"[{language}] Quote image generated")
            except Exception as e:
                logger.error(f"[{language}] Quote image generation failed: {e}")

            # Step 4: Optionally generate audio
            audio_path: Optional[str] = None
            if should_generate_audio():
                logger.info(f"[{language}] Audio generation selected")
                try:
                    script = await self.topic_gen.generate_podcast_script(
                        language, topic_title
                    )
                    audio_path = os.path.join(
                        self.image_gen.output_dir,
                        generate_unique_filename("podcast", "mp3")
                    )
                    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

                    await self.tts_service.generate_audio_from_script(
                        script, language, audio_path
                    )
                    temp_files.append(audio_path)
                    audio_generated = "true"
                    logger.info(f"[{language}] Audio generated: {audio_path}")
                except Exception as e:
                    logger.error(f"[{language}] Audio generation failed: {e}")
                    audio_generated = "failed"
                    if audio_path and os.path.exists(audio_path):
                        safe_delete(audio_path)
                        audio_path = None

            # Step 5: Send to Telegram (with safe HTML truncation)
            full_text = topic.to_telegram_html(max_length=4000)

            success_dict = await self.telegram_bot.send_topic(
                chat_id=channel_id,
                topic_text=full_text,
                article_image_path=article_img_path,
                quote_image_path=quote_img_path,
                audio_path=audio_path,
            )

            overall_success = success_dict.get("text", False)

            if not overall_success:
                logger.error(f"[{language}] Failed to send even text to Telegram")
                await self.csv_manager.append_topic(
                    language, topic_title, audio_generated
                )
                return False

            # Step 6: Log to CSV
            await self.csv_manager.append_topic(language, topic_title, audio_generated)
            logger.info(f"[{language}] Publication successful")
            return True

        except Exception as e:
            logger.exception(f"[{language}] Publication pipeline error")
            return False

        finally:
            # Cleanup temporary files
            for f in temp_files:
                if f and os.path.exists(f):
                    safe_delete(f)
                    logger.debug(f"Cleaned up: {f}")

            # Periodic cleanup of old images
            if language == get_language_codes()[0] if get_language_codes() else False:
                try:
                    deleted = self.image_gen.cleanup_temp_images(age_minutes=120)
                    if deleted:
                        logger.info(f"Cleaned up {deleted} old temporary files")
                except Exception as e:
                    logger.warning(f"Temp cleanup error: {e}")