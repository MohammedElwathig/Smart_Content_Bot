#!/usr/bin/env python3
"""
Smart Content Bot - Main Entry Point (Stable & Fortified)

Handles all service wiring, startup, keep-alive, and graceful shutdown.
Compatible with async KeyManager, GeminiClient, TopicGenerator, and TTS.
"""

import asyncio
import signal
import sys
from pathlib import Path

# Third-party (already in requirements.txt)
import aiohttp

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

# Project modules
from config.settings import settings
from src.ai.gemini_client import GeminiClient
from src.ai.key_manager import GeminiKeyManager
from src.ai.topic_generator import TopicGenerator
from src.bot.handlers import register_handlers
from src.bot.telegram_bot import TelegramBot
from src.image.image_generator import ImageGenerator
from src.scheduler.job_scheduler import JobScheduler
from src.storage.csv_manager import CSVManager
from src.storage.languages import get_languages
from src.tts.edge_tts_service import EdgeTTSService
from src.utils.logger import get_logger
from src.web.health_server import HealthServer

logger = get_logger(__name__)


async def main() -> None:
    """Main entry point: init, start, keep-alive, shutdown."""
    logger.info("=" * 50)
    logger.info("Smart Content Bot Starting...")
    logger.info(f"Environment: {settings.environment}")

    # ------------------------------------------------------------------
    # Phase 1: Validate configuration and create services
    # ------------------------------------------------------------------
    try:
        languages = get_languages()
    except Exception as e:
        logger.critical(f"Cannot load languages: {e}")
        sys.exit(1)

    if not languages:
        logger.critical("No languages configured. Exiting.")
        sys.exit(1)

    logger.info(f"Languages loaded: {', '.join(l.code for l in languages)}")

    try:
        # 1. Key Manager
        key_manager = GeminiKeyManager(settings.gemini_api_keys)
        logger.info(f"KeyManager: {key_manager.active_key_count_sync}/{len(settings.gemini_api_keys)} keys active")

        # 2. Gemini Client
        gemini_client = GeminiClient(key_manager)

        # 3. CSV Manager
        csv_manager = CSVManager()

        # 4. Topic Generator
        topic_generator = TopicGenerator(gemini_client, csv_manager)

        # 5. Image Generator
        image_generator = ImageGenerator()

        # 6. TTS Service
        tts_service = EdgeTTSService()

        # 7. Telegram Bot (NO access to .bot before initialize)
        telegram_bot = TelegramBot(settings.telegram_bot_token)

        # 8. Job Scheduler
        job_scheduler = JobScheduler(
            topic_gen=topic_generator,
            image_gen=image_generator,
            tts_service=tts_service,
            telegram_bot=telegram_bot,
            csv_manager=csv_manager,
        )

        # 9. Health Server
        health_server = HealthServer(port=settings.port)

        logger.info("All services initialized")

    except Exception as e:
        logger.critical(f"Service initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2: Register command handlers
    # ------------------------------------------------------------------
    services = {
        "telegram_bot": telegram_bot,
        "topic_generator": topic_generator,
        "csv_manager": csv_manager,
        "key_manager": key_manager,
        "job_scheduler": job_scheduler,
    }
    register_handlers(telegram_bot.application, services)
    logger.info("Command handlers registered")

    # ------------------------------------------------------------------
    # Phase 3: Start all runtime services
    # ------------------------------------------------------------------
    try:
        await health_server.start()
        logger.info("Health server started")

        await job_scheduler.start()
        logger.info("Job scheduler started")

        # Telegram bot: official way to start
        await telegram_bot.application.initialize()
        await telegram_bot.application.start()
        await telegram_bot.application.updater.start_polling()
        logger.info("Telegram polling started")
    except Exception as e:
        logger.critical(f"Failed to start services: {e}", exc_info=True)
        await _safe_shutdown(telegram_bot, job_scheduler, health_server)
        sys.exit(1)

    logger.info("Bot is fully operational and polling...")

    # ------------------------------------------------------------------
    # Phase 4: Keep-alive (self-ping) and wait for termination signal
    # ------------------------------------------------------------------
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    # Signal handling (Unix only, Render uses Linux so it's fine)
    try:
        loop.add_signal_handler(signal.SIGINT, lambda: stop_event.set())
        loop.add_signal_handler(signal.SIGTERM, lambda: stop_event.set())
    except NotImplementedError:
        logger.warning("Signal handlers not available (Windows?)")

    # Internal self-ping every 10 minutes to help prevent idle sleep
    async def self_pinger():
        url = f"http://127.0.0.1:{settings.port}/ping"
        await asyncio.sleep(30)  # let server start
        async with aiohttp.ClientSession() as session:
            while not stop_event.is_set():
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            logger.debug("Self-ping OK")
                        else:
                            logger.warning(f"Self-ping status {resp.status}")
                except Exception as e:
                    logger.debug(f"Self-ping failed (non-critical): {e}")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=600)
                except asyncio.TimeoutError:
                    continue

    ping_task = asyncio.create_task(self_pinger())

    # Main wait loop – survives most unexpected exceptions
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Phase 5: Graceful shutdown
    # ------------------------------------------------------------------
    logger.info("Shutting down...")
    ping_task.cancel()
    try:
        await ping_task
    except asyncio.CancelledError:
        pass

    await _safe_shutdown(telegram_bot, job_scheduler, health_server)
    logger.info("All services stopped. Goodbye.")
    logger.info("=" * 50)


async def _safe_shutdown(telegram_bot, job_scheduler, health_server) -> None:
    """Stop all services, ignoring any errors during shutdown."""
    # Telegram
    try:
        app = telegram_bot.application
        if hasattr(app, 'updater') and app.updater:
            await app.updater.stop()
        await app.stop()
        await app.shutdown()
        logger.info("Telegram bot stopped")
    except Exception as e:
        logger.error(f"Error stopping Telegram: {e}")

    # Scheduler
    try:
        await job_scheduler.stop()
        logger.info("Job scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")

    # Health server
    try:
        await health_server.stop()
        logger.info("Health server stopped")
    except Exception as e:
        logger.error(f"Error stopping health server: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)