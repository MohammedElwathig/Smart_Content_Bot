#!/usr/bin/env python3
"""
Smart Content Bot - Main Entry Point

Fortified for Render deployment. Handles idle sleep prevention
and survives transient failures without exiting.
"""

import asyncio
import signal
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Project imports
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
    """Application entry point with full lifecycle management."""
    logger.info("=" * 50)
    logger.info("Smart Content Bot Starting...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Log level: {settings.log_level}")

    # -------------------------------------------------------------------------
    # Validate languages
    # -------------------------------------------------------------------------
    try:
        languages = get_languages()
    except Exception as e:
        logger.critical(f"Cannot load languages: {e}")
        sys.exit(1)

    if not languages:
        logger.critical("No languages configured. Exiting.")
        sys.exit(1)

    logger.info(f"Languages: {', '.join(l.code for l in languages)}")

    # -------------------------------------------------------------------------
    # Build services
    # -------------------------------------------------------------------------
    try:
        key_manager = GeminiKeyManager(settings.gemini_api_keys)
        logger.info(f"KeyManager: {key_manager.active_key_count_sync}/{len(settings.gemini_api_keys)} keys active")

        gemini_client = GeminiClient(key_manager)
        csv_manager = CSVManager()
        topic_generator = TopicGenerator(gemini_client, csv_manager)
        image_generator = ImageGenerator()
        tts_service = EdgeTTSService()
        telegram_bot = TelegramBot(settings.telegram_bot_token)

        job_scheduler = JobScheduler(
            topic_gen=topic_generator,
            image_gen=image_generator,
            tts_service=tts_service,
            telegram_bot=telegram_bot,
            csv_manager=csv_manager,
        )

        health_server = HealthServer(port=settings.port)
        logger.info("All services initialized successfully")

    except Exception as e:
        logger.critical(f"Service initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Register handlers
    # -------------------------------------------------------------------------
    services = {
        "telegram_bot": telegram_bot,
        "topic_generator": topic_generator,
        "csv_manager": csv_manager,
        "key_manager": key_manager,
        "job_scheduler": job_scheduler,
    }
    register_handlers(telegram_bot.application, services)
    logger.info("Handlers registered")

    # -------------------------------------------------------------------------
    # Start runtime services
    # -------------------------------------------------------------------------
    try:
        await health_server.start()
        logger.info("Health server started")

        await job_scheduler.start()
        logger.info("Job scheduler started")

        await telegram_bot.application.initialize()
        await telegram_bot.application.start()
        await telegram_bot.application.updater.start_polling()
        logger.info("Telegram polling started")

    except Exception as e:
        logger.critical(f"Failed to start services: {e}", exc_info=True)
        await _safe_stop(telegram_bot, job_scheduler, health_server)
        sys.exit(1)

    logger.info("Bot is fully operational and polling...")

    # -------------------------------------------------------------------------
    # Wait for shutdown signal (with periodic self-ping)
    # -------------------------------------------------------------------------
    stop_event = asyncio.Event()

    # Define shutdown handler first
    def handle_shutdown():
        logger.info("Shutdown signal received")
        stop_event.set()

    # Register signal handlers safely
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, handle_shutdown)
        loop.add_signal_handler(signal.SIGTERM, handle_shutdown)
    except NotImplementedError:
        logger.warning("Signal handlers not supported on this platform (Windows?)")

    # Periodic self-ping every 10 minutes to help prevent Render idle sleep
    async def periodic_self_ping():
        """Ping our own health endpoint every 10 minutes."""
        await asyncio.sleep(30)  # Wait for server to be ready
        import urllib.request
        url = f"http://127.0.0.1:{settings.port}/ping"
        while not stop_event.is_set():
            try:
                # Use synchronous urllib in a thread to avoid aiohttp dep
                await asyncio.to_thread(urllib.request.urlopen, url, timeout=5)
                logger.debug("Self-ping OK")
            except Exception as e:
                logger.debug(f"Self-ping failed (non-critical): {e}")
            # Wait 10 minutes or until stop
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=600)
                break
            except asyncio.TimeoutError:
                pass

    ping_task = asyncio.create_task(periodic_self_ping())

    # Main wait loop - survives most unexpected errors
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pass  # Just checking
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            # Continue waiting

    # -------------------------------------------------------------------------
    # Graceful shutdown
    # -------------------------------------------------------------------------
    logger.info("Shutting down...")
    ping_task.cancel()
    try:
        await ping_task
    except asyncio.CancelledError:
        pass

    await _safe_stop(telegram_bot, job_scheduler, health_server)
    logger.info("All services stopped. Goodbye.")
    logger.info("=" * 50)


async def _safe_stop(telegram_bot, job_scheduler, health_server) -> None:
    """Stop all services, ignoring any errors during shutdown."""
    # Telegram bot
    try:
        app = telegram_bot.application
        if hasattr(app, 'updater') and app.updater:
            await app.updater.stop()
        await app.stop()
        await app.shutdown()
        logger.info("Telegram bot stopped")
    except Exception as e:
        logger.error(f"Error stopping Telegram bot: {e}")

    # Job scheduler
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
        logger.info("Interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal unhandled error: {e}", exc_info=True)
        sys.exit(1)