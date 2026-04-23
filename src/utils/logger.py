"""
Logging configuration module for Smart Content Bot.

Provides a pre-configured logger with console and rotating file handlers.
Supports redaction of sensitive information and configurable log levels.
"""

import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

# Constants
LOG_DIR: str = "logs"
LOG_FILE: str = os.path.join(LOG_DIR, "bot.log")
MAX_LOG_SIZE: int = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT: int = 3
DEFAULT_LOG_LEVEL: str = "INFO"

# Regex patterns for redacting sensitive information
# Gemini API key format: "AIza" followed by 35 alphanumeric, dash, or underscore
GEMINI_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z\-_]{35}")
# Telegram bot token format: <bot_id>:<35-character hash>
TELEGRAM_TOKEN_PATTERN = re.compile(r"\d{9,10}:[0-9A-Za-z\-_]{35}")


class RedactingFormatter(logging.Formatter):
    """
    Custom formatter that redacts sensitive information from log messages.

    Redacts Gemini API keys and Telegram bot tokens automatically.
    """

    def format(self, record: logging.LogRecord) -> str:
        original_msg = super().format(record)
        # Apply redaction for known sensitive patterns
        redacted_msg = GEMINI_KEY_PATTERN.sub(
            "***REDACTED_GEMINI_KEY***", original_msg
        )
        redacted_msg = TELEGRAM_TOKEN_PATTERN.sub(
            "***REDACTED_TELEGRAM_TOKEN***", redacted_msg
        )
        return redacted_msg


def _setup_logging() -> None:
    """
    Internal function to configure logging handlers and formatters.
    Called automatically on module import.
    """
    # Ensure log directory exists
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except OSError as e:
        print(
            f"Warning: Could not create log directory '{LOG_DIR}': {e}. "
            "File logging disabled.",
            file=sys.stderr,
        )

    # Determine log level from environment directly.
    # We intentionally read os.environ instead of importing settings
    # to avoid circular imports during early initialization.
    log_level_str = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    try:
        console_level = getattr(logging, log_level_str)
    except AttributeError:
        console_level = logging.INFO
        print(
            f"Warning: Invalid LOG_LEVEL '{log_level_str}'. Using INFO.",
            file=sys.stderr,
        )

    # Create formatter
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    date_format = "%Y-%m-%dT%H:%M:%S"
    formatter = RedactingFormatter(fmt=log_format, datefmt=date_format)

    # Get the project-specific logger (not root)
    project_logger = logging.getLogger("smart_content_bot")
    project_logger.setLevel(logging.DEBUG)  # Let handlers filter
    project_logger.propagate = False

    # Add a NullHandler as fallback to avoid "No handlers could be found" warnings
    # if all handlers fail (will be removed after adding real handlers).
    project_logger.addHandler(logging.NullHandler())

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    project_logger.addHandler(console_handler)

    # Rotating file handler (optional, may fail gracefully)
    try:
        file_handler = RotatingFileHandler(
            filename=LOG_FILE,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # Always debug to file
        file_handler.setFormatter(formatter)
        project_logger.addHandler(file_handler)
    except Exception as e:
        print(
            f"Warning: Could not set up file logging: {e}. "
            "Continuing with console only.",
            file=sys.stderr,
        )

    # Remove the temporary NullHandler now that we have real handlers
    for handler in project_logger.handlers[:]:
        if isinstance(handler, logging.NullHandler):
            project_logger.removeHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger instance for the given module name.

    Args:
        name: Usually __name__ of the calling module. If None, returns
              the base project logger.

    Returns:
        Configured logger instance.
    """
    if name is None:
        return logging.getLogger("smart_content_bot")
    return logging.getLogger(f"smart_content_bot.{name}")


# Run setup on import
_setup_logging()