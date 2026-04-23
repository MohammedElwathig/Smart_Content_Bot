"""
General purpose utility functions for the Smart Content Bot.

Provides reusable, stateless helpers for date/time, file operations,
text processing, and more. All functions are pure and import-safe.
"""

import asyncio
import datetime
import os
import random
import re
import time
import uuid
from typing import Any, Awaitable, Callable, List, Optional, TypeVar, Union

# Type variable for async retry decorator
T = TypeVar("T")


# =============================================================================
# Date and Time Utilities
# =============================================================================

def utc_now_iso() -> str:
    """
    Return current UTC time in ISO 8601 format with seconds precision and 'Z' suffix.

    Example:
        "2026-04-22T14:30:45Z"

    Returns:
        ISO formatted UTC datetime string.
    """
    # Python 3.11+ has datetime.UTC
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_now_timestamp() -> float:
    """
    Return current UTC timestamp (Unix epoch seconds).

    Returns:
        Float representing seconds since epoch.
    """
    return time.time()


# =============================================================================
# File System Utilities (Ephemeral-Safe)
# =============================================================================

def safe_delete(file_path: str) -> bool:
    """
    Safely delete a file, ignoring "file not found" errors.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        True if file was deleted or did not exist; False on permission error.
    """
    try:
        os.remove(file_path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def ensure_directory(path: str) -> None:
    """
    Create a directory and all parent directories if they don't exist.

    Args:
        path: Directory path to create.

    Raises:
        OSError: If creation fails due to permissions or other issues.
    """
    os.makedirs(path, exist_ok=True)


def generate_unique_filename(prefix: str, extension: str) -> str:
    """
    Generate a unique filename for temporary media files.

    Args:
        prefix: Descriptive prefix (e.g., "topic_image").
        extension: File extension without dot (e.g., "png").

    Returns:
        Unique filename like "prefix_20260422T143045Z_123456_abc123.png".
    """
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    random_str = uuid.uuid4().hex[:6]
    return f"{prefix}_{timestamp}_{random_str}.{extension}"


# =============================================================================
# Text and String Utilities
# =============================================================================

def chunk_text(text: str, max_length: int = 4096) -> List[str]:
    """
    Split long text into chunks suitable for Telegram messages.

    Tries to preserve paragraph boundaries, then sentence boundaries,
    and finally falls back to hard splitting at max_length.

    Args:
        text: The text to split.
        max_length: Maximum length of each chunk (default Telegram limit).

    Returns:
        List of text chunks, each ≤ max_length.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    # First try splitting by double newline (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current = ""

    for para in paragraphs:
        # If adding this paragraph fits, append it
        if len(current) + len(para) + (2 if current else 0) <= max_length:
            current = current + "\n\n" + para if current else para
        else:
            # Paragraph doesn't fit as a whole
            if current:
                chunks.append(current)
                current = ""

            # If the paragraph itself is too long, split it further
            if len(para) > max_length:
                # Try splitting by sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    if len(sent) > max_length:
                        # Sentence too long, force split at max_length
                        for i in range(0, len(sent), max_length):
                            chunks.append(sent[i:i+max_length])
                    else:
                        # Sentence fits, but maybe not in current chunk
                        if len(current) + len(sent) + 1 <= max_length:
                            current = current + " " + sent if current else sent
                        else:
                            chunks.append(current)
                            current = sent
            else:
                # Paragraph fits by itself
                current = para

    if current:
        chunks.append(current)

    # Remove any empty strings that might have slipped through
    return [c for c in chunks if c]


def redact_sensitive(text: str) -> str:
    """
    Remove or mask sensitive information like API keys and tokens.

    Args:
        text: The string potentially containing secrets.

    Returns:
        String with secrets replaced by "***REDACTED***".
    """
    # Gemini API key pattern: AIza followed by 35 alphanumeric, dash, underscore
    text = re.sub(r"AIza[0-9A-Za-z\-_]{35}", "***REDACTED_GEMINI_KEY***", text)
    # Telegram bot token pattern: 8-10 digits, colon, 35+ character hash
    text = re.sub(r"\d{8,10}:[0-9A-Za-z\-_]{35,}", "***REDACTED_TELEGRAM_TOKEN***", text)
    return text


# =============================================================================
# Data Structure Utilities
# =============================================================================

def parse_comma_separated_list(value: str) -> List[str]:
    """
    Convert a comma-separated string into a list of trimmed, non-empty strings.

    Args:
        value: Comma-separated string (e.g., "a, b, , c").

    Returns:
        List of cleaned strings (e.g., ["a", "b", "c"]).
    """
    return [item.strip() for item in value.split(",") if item.strip()]


# =============================================================================
# Randomization and Probability
# =============================================================================

def should_generate_audio(denominator: int) -> bool:
    """
    Return True with probability 1/denominator.

    Args:
        denominator: The N in 1/N chance. Must be positive.

    Returns:
        True if random event occurs, False otherwise.
    """
    if denominator <= 0:
        denominator = 1
    return random.randint(1, denominator) == 1


# =============================================================================
# Async Retry Utility
# =============================================================================

async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[type, tuple] = Exception,
    **kwargs: Any,
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async callable to retry.
        *args: Positional arguments for func.
        max_retries: Maximum number of attempts (default 3).
        delay: Initial delay between retries in seconds.
        backoff_factor: Multiplier for delay after each failure.
        exceptions: Exception type(s) to catch and retry.
        **kwargs: Keyword arguments for func.

    Returns:
        Result of func(*args, **kwargs).

    Raises:
        The last exception after all retries are exhausted.
    """
    last_exception: Optional[Exception] = None
    current_delay = delay

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(current_delay)
            current_delay *= backoff_factor

    # Should never reach here, but for type safety
    raise last_exception  # type: ignore