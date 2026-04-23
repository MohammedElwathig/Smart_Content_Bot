"""
Language configuration loader for Smart Content Bot.

Loads and validates language-to-channel mappings from data/languages.csv.
"""

import asyncio
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# CSV file path relative to project root
CSV_PATH = os.path.join("data", "languages.csv")

# Lock for thread-safe reload operations (used in async context)
_reload_lock = asyncio.Lock()


class LanguageConfigError(Exception):
    """Raised when the languages.csv file is missing or invalid."""
    pass


@dataclass(frozen=True)
class LanguageConfig:
    """Immutable configuration for a single language."""
    code: str          # Language code (e.g., "ar", "en-US")
    channel_id: int    # Telegram channel ID
    name: str          # Human-readable name (e.g., "Arabic")


# Module-level cache
_LANGUAGES: List[LanguageConfig] = []
_CHANNEL_MAP: Dict[str, int] = {}
_NAME_MAP: Dict[str, str] = {}


def _load_languages() -> List[LanguageConfig]:
    """
    Load and validate the languages.csv file.

    Returns:
        List of validated LanguageConfig objects.

    Raises:
        LanguageConfigError: If file missing, columns invalid, or data malformed.
    """
    if not os.path.exists(CSV_PATH):
        raise LanguageConfigError(
            f"Language configuration file not found: {CSV_PATH}. "
            "Please create it with columns: language_code,channel_id,language_name"
        )

    languages = []
    required_columns = {"language_code", "channel_id", "language_name"}

    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Validate header
            if not reader.fieldnames:
                raise LanguageConfigError("CSV header is empty")
            missing = required_columns - set(reader.fieldnames)
            if missing:
                raise LanguageConfigError(
                    f"CSV missing required columns: {missing}. "
                    f"Found: {reader.fieldnames}"
                )

            for row_num, row in enumerate(reader, start=2):
                code = row.get("language_code", "").strip()
                channel_str = row.get("channel_id", "").strip()
                name = row.get("language_name", "").strip()

                # Validate language_code (non-empty)
                if not code:
                    logger.warning(f"Row {row_num}: empty language_code, skipping")
                    continue
                # Note: Language codes can be longer than 2 characters (e.g., "en-US").
                # We accept any non-empty string.

                # Validate channel_id
                if not channel_str:
                    logger.warning(f"Row {row_num}: empty channel_id, skipping")
                    continue
                try:
                    channel_id = int(channel_str)
                except ValueError:
                    raise LanguageConfigError(
                        f"Row {row_num}: channel_id '{channel_str}' is not a valid integer"
                    )

                # Validate language_name (fallback to code if missing)
                if not name:
                    name = code.upper()
                    logger.debug(f"Row {row_num}: empty language_name, using '{name}'")

                languages.append(LanguageConfig(code=code, channel_id=channel_id, name=name))

    except csv.Error as e:
        raise LanguageConfigError(f"CSV parsing error: {e}")

    if not languages:
        raise LanguageConfigError("No valid language configurations found in CSV")

    logger.info(f"Loaded {len(languages)} language(s): {[l.code for l in languages]}")
    return languages


# Load configuration at module import
try:
    _LANGUAGES = _load_languages()
    _CHANNEL_MAP = {lang.code: lang.channel_id for lang in _LANGUAGES}
    _NAME_MAP = {lang.code: lang.name for lang in _LANGUAGES}
except LanguageConfigError as e:
    logger.critical(f"Language configuration error: {e}")
    raise
except Exception as e:
    logger.exception("Unexpected error loading languages")
    raise


# Public accessor functions
def get_languages() -> List[LanguageConfig]:
    """
    Return a shallow copy of the list of all configured languages.

    Returns:
        List of LanguageConfig objects.
    """
    return list(_LANGUAGES)


def get_channel_id(language_code: str) -> Optional[int]:
    """
    Get the Telegram channel ID for a given language code.

    Args:
        language_code: Language code (e.g., "ar").

    Returns:
        Channel ID as int, or None if language not found.
    """
    return _CHANNEL_MAP.get(language_code.lower())


def get_language_codes() -> List[str]:
    """
    Return list of all configured language codes.

    Returns:
        List of language code strings.
    """
    return list(_CHANNEL_MAP.keys())


def get_language_names() -> Dict[str, str]:
    """
    Return a copy of the mapping of language codes to human-readable names.

    Returns:
        Dictionary like {"ar": "Arabic", "en": "English"}.
    """
    return dict(_NAME_MAP)


def is_language_supported(language_code: str) -> bool:
    """
    Check if a language code is configured.

    Args:
        language_code: Language code (e.g., "ar").

    Returns:
        True if language exists in configuration.
    """
    return language_code.lower() in _CHANNEL_MAP


async def reload_languages() -> None:
    """
    Force reload of language configuration from CSV.

    Useful for admin commands that update languages.csv at runtime.
    This function is async-safe and uses a lock to prevent concurrent reloads.

    Raises:
        LanguageConfigError: If reload fails.
    """
    global _LANGUAGES, _CHANNEL_MAP, _NAME_MAP
    async with _reload_lock:
        try:
            new_languages = _load_languages()
            _LANGUAGES = new_languages
            _CHANNEL_MAP = {lang.code: lang.channel_id for lang in new_languages}
            _NAME_MAP = {lang.code: lang.name for lang in new_languages}
            logger.info("Language configuration reloaded")
        except Exception as e:
            logger.exception("Failed to reload language configuration")
            raise