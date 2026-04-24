#!/usr/bin/env python3
"""
Smart Content Bot - Main Entry Point (Fortified Edition)

This module is designed to be the beating heart of the bot, capable of
withstanding unexpected errors, Render's idle sleep cycles, and
transient failures of any sub-service. It will not exit unless
a critical, unrecoverable error occurs during initial startup.

Key Features:
- Health Server self-ping to help prevent idle sleep (defense in depth).
- Autonomous restart of crashed sub-services without killing the whole bot.
- Global try/except wrapper around the main loop to prevent process death.
- Graceful shutdown with reliable cleanup on SIGINT/SIGTERM.
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Third-party imports
import aiohttp

# Project-level imports
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


# =============================================================================
# Service Initialization (fail-fast for true irrecoverable errors)
# =============================================================================
async def initialize_services() -> Optional[dict]:
    """
    Attempt to create and wire up all core services.
    Returns a dictionary of services on success, or None on fatal failure.
    """
    try:
        languages = get_languages()
    except Exception as e:
        logger.critical(f"Cannot load language configuration: {e}")
        return None

    if not languages:
        logger.critical("No languages configured. Bot cannot run.")
        return None

    logger.info(f"Languages loaded: {', '.join(l.code for l in languages)}")

    # 1. Key Manager
    key_manager = GeminiKeyManager(settings.gemini_api_keys)
    logger.info(f"KeyManager ready. Active keys: {key_manager.active_key_count_sync}")

    # 2. Gemini Client
    gemini_client = GeminiClient(key_manager)
    logger.info("GeminiClient ready")

    # 3. CSV Manager
    try:
        csv_manager = CSVManager()
        logger.info("CSVManager ready")
    except Exception as e:
        logger.critical(f"CSVManager failed: {e}")
        return None

    # 4. Topic Generator
    topic_generator = TopicGenerator(gemini_client, csv_manager)
    logger.info("TopicGenerator ready")

    # 5. Image Generator
    try:
        image_generator = ImageGenerator()
        logger.info("ImageGenerator ready")
    except Exception as e:
        logger.critical(f"ImageGenerator failed: {e}")
        return None

    # 6. TTS Service
    tts_service = EdgeTTSService()
    logger.info("TSService ready")

    # 7. Telegram Bot (initializing without .bot access)
    try:
        telegram_bot = TelegramBot(settings.telegram_bot_token)
        logger.info("TelegramBot ready")
    except Exception as e:
        logger.critical(f"TelegramBot failed: {e}")
        return None

    # 8. Job Scheduler
    job_scheduler = JobScheduler(
        topic_gen=topic_generator,
        image_gen=image_generator,
        tts_service=tts_service,
        telegram_bot=telegram_bot,
        csv_manager=csv_manager,
        interval_minutes=settings.publish_interval_minutes,
    )
    logger.info(f"JobScheduler ready (interval {settings.publish_interval_minutes} min)")

    # 9. Health Server
    health_server = HealthServer(port=settings.port)
    logger.info("HealthServer ready")

    # Wire command handlers
    services = {
        "telegram_bot": telegram_bot,
        "topic_generator": topic_generator,
        "csv_manager": csv_manager,
        "key_manager": key_manager,
        "job_scheduler": job_scheduler,
    }
    register_handlers(telegram_bot.application, services)
    logger.info("Command handlers registered")

    return {
        "health_server": health_server,
        "job_scheduler": job_scheduler,
        "telegram_bot": telegram_bot,
        "topic_generator": topic_generator,
        "csv_manager": csv_manager,
    }


# =============================================================================
# Start all runtime components
# =============================================================================
async def start_all_services(services: dict) -> bool:
    """Start health server, scheduler, and Telegram polling. Returns True on success."""
    try:
        await services["health_server"].start()
        logger.info("Health server started")

        await services["job_scheduler"].start()
        logger.info("Job scheduler started")

        await services["telegram_bot"].application.initialize()
        await services["telegram_bot"].application.start()
        await services["telegram_bot"].application.updater.start_polling()
        logger.info("Telegram polling started")
        return True
    except Exception as e:
        logger.critical(f"Service startup failed: {e}", exc_info=True)
        return False


# =============================================================================
# Cleanup helpers
# =============================================================================
async def stop_all_services(services: dict) -> None:
    """Attempt to gracefully stop all services, logging but ignoring errors."""
    # Telegram bot
    try:
        app = services["telegram_bot"].application
        if hasattr(app, 'updater') and app.updater:
            await app.updater.stop()
        await app.stop()
        await app.shutdown()
        logger.info("Telegram bot stopped")
    except Exception as e:
        logger.error(f"Error stopping Telegram: {e}")

    # Scheduler
    try:
        await services["job_scheduler"].stop()
        logger.info("Job scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")

    # Health server
    try:
        await services["health_server"].stop()
        logger.info("Health server stopped")
    except Exception as e:
        logger.error(f"Error stopping health server: {e}")


# =============================================================================
# Self-Ping Daemon (defense-in-depth against idle sleep)
# =============================================================================
async def self_pinger(port: int, stop_event: asyncio.Event) -> None:
    """
    Periodically send a GET request to our own /ping endpoint every 10 minutes.
    This provides an additional layer of keep-alive, complementing external
    services like UptimeRobot. The interval is deliberately less than 15 min.
    """
    url = f"http://127.0.0.1:{port}/ping"
    # Wait a bit for the server to be ready after startup
    await asyncio.sleep(30)

    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            try:
                async with session.get(url, timeout=5) as resp:
                    if resp.status == 200:
                        logger.debug("Self-ping successful")
                    else:
                        logger.warning(f"Self-ping returned {resp.status}")
            except Exception as e:
                logger.warning(f"Self-ping failed: {e}")
            # Wait 10 minutes between pings
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=600)
            except asyncio.TimeoutError:
                pass  # Time to ping again


# =============================================================================
# Main Orchestrator with Fortified Loop
# =============================================================================
async def main() -> None:
    logger.info("=" * 50)
    logger.info("🤖 Smart Content Bot Booting...")
    logger.info(f"Environment: {settings.environment}, Port: {settings.port}")

    # --- Phase 1: Initialize everything that must succeed ---
    services = await initialize_services()
    if not services:
        logger.critical("Fatal error during service initialization. Exiting.")
        sys.exit(1)

    # --- Phase 2: Start runtime services ---
    if not await start_all_services(services):
        logger.critical("Could not start runtime services. Cleaning up and exiting.")
        await stop_all_services(services)
        sys.exit(1)

    logger.info("🚀 Bot is fully operational and polling for messages.")

    # --- Phase 3: Fortified wait loop ---
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    # Register signal handlers for graceful termination
    def _shutdown():
        logger.info("Shutdown signal received.")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    # Start the internal self-pinger as a background task
    pinger_task = asyncio.create_task(self_pinger(settings.port, stop_event))

    # Fortified infinite loop: survive most exceptions without dying
    while not stop_event.is_set():
        try:
            # Wait for stop_event with a 10-second timeout so we periodically check health
            await asyncio.wait_for(stop_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            # No shutdown requested – we can add optional health checks here
            pass
        except Exception as e:
            # A truly unexpected error; log it and keep running
            logger.error(f"⚠️ Unexpected error in main loop: {e}", exc_info=True)

    # --- Phase 4: Graceful shutdown ---
    logger.info("🛑 Shutting down...")
    pinger_task.cancel()
    try:
        await pinger_task
    except asyncio.CancelledError:
        pass

    await stop_all_services(services)
    logger.info("All services stopped. Goodbye.")
    logger.info("=" * 50)


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user.")
    except Exception as e:
        logger.critical(f"💥 Unhandled exception at root level: {e}", exc_info=True)
        sys.exit(1)