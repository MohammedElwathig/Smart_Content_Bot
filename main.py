#!/usr/bin/env python3
"""
Smart Content Bot - Main Entry Point

This module serves as the composition root, wiring all services together,
starting background tasks, and managing graceful shutdown.
"""

import asyncio
import signal
import sys
from pathlib import Path

# Add project root to Python path (for Render compatibility)
sys.path.insert(0, str(Path(__file__).parent))

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
    """Orchestrate application startup, runtime, and graceful shutdown."""
    logger.info("=" * 50)
    logger.info("Smart Content Bot Starting...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Log level: {settings.log_level}")

    # Log redacted configuration for debugging
    safe_config = settings.redacted_dict()
    logger.debug(f"Configuration: {safe_config}")

    # -------------------------------------------------------------------------
    # Validate language configuration
    # -------------------------------------------------------------------------
    try:
        languages = get_languages()
    except Exception as e:
        logger.critical(f"Failed to load language configuration: {e}")
        sys.exit(1)

    if not languages:
        logger.critical("No languages configured in data/languages.csv. Exiting.")
        sys.exit(1)

    logger.info(f"Loaded {len(languages)} language(s): {', '.join([l.code for l in languages])}")

    # -------------------------------------------------------------------------
    # Instantiate services in dependency order
    # -------------------------------------------------------------------------

    # 1. Key Manager
    try:
        key_manager = GeminiKeyManager(settings.gemini_api_keys)
        logger.info(
            f"GeminiKeyManager initialized: {key_manager.active_key_count_sync}/{len(settings.gemini_api_keys)} keys active"
        )
    except Exception as e:
        logger.critical(f"Failed to initialize GeminiKeyManager: {e}")
        sys.exit(1)

    # 2. Gemini Client
    gemini_client = GeminiClient(key_manager)
    logger.info("GeminiClient initialized")

    # 3. CSV Manager
    try:
        csv_manager = CSVManager()
        logger.info("CSVManager initialized")
    except Exception as e:
        logger.critical(f"Failed to initialize CSVManager: {e}")
        sys.exit(1)

    # 4. Topic Generator
    topic_generator = TopicGenerator(gemini_client, csv_manager)
    logger.info("TopicGenerator initialized")

    # 5. Image Generator
    try:
        image_generator = ImageGenerator()
        logger.info("ImageGenerator initialized")
    except Exception as e:
        logger.critical(f"Failed to initialize ImageGenerator: {e}")
        sys.exit(1)

    # 6. TTS Service
    tts_service = EdgeTTSService()
    logger.info("EdgeTTSService initialized")

    # 7. Telegram Bot (do not access bot.username before initialize)
    try:
        telegram_bot = TelegramBot(settings.telegram_bot_token)
        logger.info("TelegramBot initialized")
    except Exception as e:
        logger.critical(f"Failed to initialize TelegramBot: {e}")
        sys.exit(1)

    # 8. Job Scheduler
    job_scheduler = JobScheduler(
        topic_gen=topic_generator,
        image_gen=image_generator,
        tts_service=tts_service,
        telegram_bot=telegram_bot,
        csv_manager=csv_manager,
        interval_minutes=settings.publish_interval_minutes,
    )
    logger.info(f"JobScheduler initialized (interval: {settings.publish_interval_minutes} min)")

    # 9. Health Server
    health_server = HealthServer(port=settings.port)
    logger.info(f"HealthServer initialized on port {settings.port}")

    # -------------------------------------------------------------------------
    # Register Telegram command handlers
    # -------------------------------------------------------------------------
    services = {
        "telegram_bot": telegram_bot,
        "topic_generator": topic_generator,
        "csv_manager": csv_manager,
        "key_manager": key_manager,
        "job_scheduler": job_scheduler,
    }
    register_handlers(telegram_bot.application, services)
    logger.info("Command handlers registered")

    # -------------------------------------------------------------------------
    # Start all background services
    # -------------------------------------------------------------------------
    try:
        await health_server.start()
        logger.info("Health server started")

        await job_scheduler.start()
        logger.info("Job scheduler started")

        await telegram_bot.application.initialize()
        await telegram_bot.application.start()
        await telegram_bot.application.updater.start_polling()
        logger.info("Telegram bot polling started")

    except Exception as e:
        logger.critical(f"Failed to start services: {e}")
        await cleanup_services(health_server, job_scheduler, telegram_bot)
        sys.exit(1)

    logger.info("All services started successfully. Bot is running and polling...")

    # -------------------------------------------------------------------------
    # Wait for shutdown signal
    # -------------------------------------------------------------------------
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Shutdown signal received.")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    await stop_event.wait()

    # -------------------------------------------------------------------------
    # Graceful shutdown
    # -------------------------------------------------------------------------
    logger.info("Initiating graceful shutdown...")
    await cleanup_services(health_server, job_scheduler, telegram_bot)

    logger.info("All services stopped. Exiting gracefully.")
    logger.info("=" * 50)


async def cleanup_services(
    health_server: HealthServer,
    job_scheduler: JobScheduler,
    telegram_bot: TelegramBot,
) -> None:
    """
    Attempt to gracefully stop all services, logging any errors.
    """
    try:
        if telegram_bot.application.updater:
            await telegram_bot.application.updater.stop()
        await telegram_bot.application.stop()
        await telegram_bot.application.shutdown()
        logger.info("Telegram bot stopped")
    except Exception as e:
        logger.error(f"Error stopping Telegram bot: {e}")

    try:
        await job_scheduler.stop()
        logger.info("Job scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping job scheduler: {e}")

    try:
        await health_server.stop()
        logger.info("Health server stopped")
    except Exception as e:
        logger.error(f"Error stopping health server: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
        sys.exit(1)