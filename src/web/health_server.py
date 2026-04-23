"""
Lightweight HTTP health check server for Render deployment.

This module is CRITICAL for the stability of the bot on Render's free tier.
It serves two essential purposes:

1. **Render Health Check**: Render sends a request every 30 seconds to verify
   the service is alive. Without a quick response, Render assumes the service
   is dead and kills/redeploys it.

2. **Keep-Alive (Anti-Sleep)**: Free Render services sleep after 15 minutes of
   no incoming HTTP traffic. By setting up an external service (Uptime Robot,
   cron-job.org) to ping this server every 5 minutes, we prevent the bot from
   ever sleeping.

This server runs in the same asyncio event loop as the Telegram bot, so it
does not interfere with polling or scheduled jobs.
"""

from typing import Optional

from aiohttp import web

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HealthServer:
    """
    Minimal aiohttp server for health check and keep-alive endpoints.

    Routes:
        GET /      -> Friendly status message
        GET /ping  -> Returns "pong" (primary health check endpoint)
        GET /health -> Same as /ping (alternative name for monitoring tools)
    """

    def __init__(self, port: Optional[int] = None) -> None:
        """
        Initialize the health server.

        Args:
            port: Port to listen on. If None, uses settings.PORT (default 10000).
        """
        self.port = port if port is not None else settings.port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

        # Register routes
        self.app.router.add_get("/", self.root_handler)
        self.app.router.add_get("/ping", self.ping_handler)
        self.app.router.add_get("/health", self.ping_handler)  # Alias for /ping

        logger.info(f"HealthServer configured for port {self.port}")

    async def ping_handler(self, request: web.Request) -> web.Response:
        """
        Handle /ping and /health requests.

        Responds immediately with 'pong' to satisfy health checks.
        This handler must be extremely lightweight to avoid timeouts.
        """
        logger.debug(f"Health check from {request.remote}")
        return web.Response(text="pong", content_type="text/plain")

    async def root_handler(self, request: web.Request) -> web.Response:
        """
        Handle root path - returns a friendly status message.
        Useful for manual browser checks.
        """
        return web.Response(
            text="🤖 Smart Content Bot is online.\n\nEndpoints:\n  /ping  - Health check\n  /health - Alias for /ping",
            content_type="text/plain",
        )

    async def start(self) -> None:
        """
        Start the HTTP server on 0.0.0.0:port.

        Binds to all interfaces (0.0.0.0) so Render's health checker can reach it.

        Raises:
            OSError: If the port is already in use or cannot be bound.
        """
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self._site = web.TCPSite(self.runner, "0.0.0.0", self.port)
            await self._site.start()
            logger.info(f"✅ Health server started successfully on port {self.port}")
        except OSError as e:
            logger.critical(f"❌ Failed to bind health server to port {self.port}: {e}")
            raise
        except Exception as e:
            logger.critical(f"❌ Unexpected error starting health server: {e}")
            raise

    async def stop(self) -> None:
        """
        Gracefully shutdown the server and release the port.
        """
        if self.runner:
            try:
                await self.runner.cleanup()
                logger.info("✅ Health server stopped gracefully")
            except Exception as e:
                logger.error(f"⚠️ Error during health server shutdown: {e}")