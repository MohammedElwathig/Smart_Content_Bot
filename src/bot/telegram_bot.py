"""
Telegram Bot wrapper for Smart Content Bot.

Provides a clean async interface for sending text, photos, and audio
to Telegram channels, with built-in retry logic and error handling.
"""

import asyncio
import os
from typing import Any, Callable, Dict, Optional

from telegram import Bot, InputFile
from telegram.error import NetworkError, TelegramError, TimedOut
from telegram.ext import Application, ApplicationBuilder

from config.settings import settings
from src.utils.helpers import safe_delete, retry_async
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TelegramBot:
    """
    Async wrapper around python-telegram-bot Application.
    Handles sending messages and media with automatic retries.
    """

    def __init__(self, token: Optional[str] = None) -> None:
        """
        Initialize the Telegram bot.

        Args:
            token: Bot token from BotFather. If None, uses settings.TELEGRAM_BOT_TOKEN.
        """
        self.token = token or settings.telegram_bot_token
        self.logger = logger

        # Build application with sensible timeouts
        self.app: Application = (
            ApplicationBuilder()
            .token(self.token)
            .connect_timeout(30)
            .read_timeout(30)
            .write_timeout(30)
            .build()
        )
        self.bot: Bot = self.app.bot

        self.logger.info(f"TelegramBot initialized for @{self.bot.username}")

    @property
    def application(self) -> Application:
        """Expose the Application instance for handler registration."""
        return self.app

    async def _retry_telegram(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        **kwargs,
    ) -> Any:
        """
        Retry a Telegram API call with exponential backoff on network issues.

        Uses the general `retry_async` helper, catching only Telegram-specific
        retryable exceptions.

        Args:
            func: Async callable to execute.
            max_retries: Maximum number of attempts.
            *args, **kwargs: Arguments passed to func.

        Returns:
            Result of func.

        Raises:
            TelegramError: For non-retryable Telegram errors.
            Exception: After all retries exhausted.
        """
        return await retry_async(
            func,
            *args,
            max_retries=max_retries,
            delay=1.0,
            backoff_factor=2.0,
            exceptions=(NetworkError, TimedOut),
            **kwargs,
        )

    async def send_text(
        self,
        chat_id: int,
        text: str,
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
    ) -> bool:
        """
        Send a text message to a chat.

        Args:
            chat_id: Telegram chat/channel ID.
            text: Message text (HTML formatted by default).
            parse_mode: 'HTML' or 'MarkdownV2'.
            disable_web_page_preview: Disable link previews.

        Returns:
            True if sent successfully, False otherwise.
        """
        try:
            await self._retry_telegram(
                self.bot.send_message,
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
            )
            self.logger.debug(f"Sent text to {chat_id}: {text[:50]}...")
            return True
        except TelegramError as e:
            self.logger.error(f"Telegram error sending text to {chat_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending text to {chat_id}: {e}")
            return False

    async def send_photo(
        self,
        chat_id: int,
        photo_path: str,
        caption: Optional[str] = None,
        parse_mode: str = "HTML",
        delete_after: bool = True,
    ) -> bool:
        """
        Send a photo to a chat.

        Args:
            chat_id: Telegram chat ID.
            photo_path: Path to the image file.
            caption: Optional caption text.
            parse_mode: Parse mode for caption.
            delete_after: Whether to delete the local file after sending.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not os.path.exists(photo_path):
            self.logger.error(f"Photo file not found: {photo_path}")
            return False

        try:
            with open(photo_path, "rb") as f:
                await self._retry_telegram(
                    self.bot.send_photo,
                    chat_id=chat_id,
                    photo=InputFile(f),
                    caption=caption,
                    parse_mode=parse_mode,
                )
            self.logger.debug(f"Sent photo to {chat_id}: {os.path.basename(photo_path)}")
            return True
        except TelegramError as e:
            self.logger.error(f"Telegram error sending photo to {chat_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending photo to {chat_id}: {e}")
            return False
        finally:
            if delete_after and os.path.exists(photo_path):
                safe_delete(photo_path)

    async def send_audio(
        self,
        chat_id: int,
        audio_path: str,
        title: Optional[str] = None,
        performer: str = "Smart Content Bot",
        duration: Optional[int] = None,
        delete_after: bool = True,
    ) -> bool:
        """
        Send an audio file (MP3) to a chat.

        Args:
            chat_id: Telegram chat ID.
            audio_path: Path to the audio file.
            title: Audio title.
            performer: Performer name.
            duration: Duration in seconds (optional).
            delete_after: Whether to delete the local file after sending.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio file not found: {audio_path}")
            return False

        try:
            with open(audio_path, "rb") as f:
                await self._retry_telegram(
                    self.bot.send_audio,
                    chat_id=chat_id,
                    audio=InputFile(f),
                    title=title,
                    performer=performer,
                    duration=duration,
                )
            self.logger.debug(f"Sent audio to {chat_id}: {os.path.basename(audio_path)}")
            return True
        except TelegramError as e:
            self.logger.error(f"Telegram error sending audio to {chat_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending audio to {chat_id}: {e}")
            return False
        finally:
            if delete_after and os.path.exists(audio_path):
                safe_delete(audio_path)

    async def send_topic(
        self,
        chat_id: int,
        topic_text: str,
        article_image_path: Optional[str] = None,
        quote_image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Send a complete topic package: text, article image, quote image, and optional audio.

        This method tolerates missing images/audio gracefully by skipping them.

        Args:
            chat_id: Telegram channel ID.
            topic_text: Full article text (HTML formatted).
            article_image_path: Path to article image (optional).
            quote_image_path: Path to quote image (optional).
            audio_path: Optional path to podcast audio file.

        Returns:
            Dictionary with success status for each component.
        """
        results = {
            "text": False,
            "article_image": False,
            "quote_image": False,
            "audio": False,
        }

        # 1. Send text (required)
        results["text"] = await self.send_text(chat_id, topic_text)

        # 2. Send article image (optional)
        if article_image_path:
            results["article_image"] = await self.send_photo(
                chat_id, article_image_path, delete_after=True
            )
        else:
            results["article_image"] = True  # skipped successfully

        # 3. Send quote image (optional)
        if quote_image_path:
            results["quote_image"] = await self.send_photo(
                chat_id, quote_image_path, delete_after=True
            )
        else:
            results["quote_image"] = True

        # 4. Send audio (optional)
        if audio_path:
            results["audio"] = await self.send_audio(
                chat_id, audio_path, delete_after=True
            )

        self.logger.info(
            f"Topic package sent to {chat_id}: text={results['text']}, "
            f"article_img={results['article_image']}, quote_img={results['quote_image']}, "
            f"audio={results['audio']}"
        )
        return results

    async def stop(self) -> None:
        """
        Gracefully stop the bot and release resources.
        """
        self.logger.info("Stopping Telegram bot...")
        try:
            await self.app.stop()
            await self.app.shutdown()
            self.logger.info("Telegram bot stopped")
        except Exception as e:
            self.logger.error(f"Error during bot shutdown: {e}")