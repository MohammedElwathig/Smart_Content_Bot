"""
CSV-based publication history manager for Smart Content Bot.

Provides async-safe methods to append and query the topics log.
"""

import asyncio
import csv
import os
from typing import Any, Dict, List, Optional

from src.utils.helpers import ensure_directory, utc_now_iso
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CSVManager:
    """
    Manages the topics_log.csv file with async-safe operations.
    Uses a lock to prevent concurrent read/write corruption.
    """

    def __init__(self, file_path: str = "data/topics_log.csv") -> None:
        """
        Initialize the CSV manager.

        Args:
            file_path: Path to the CSV log file.
        """
        self.file_path = file_path
        self._lock = asyncio.Lock()
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create directory and CSV file with headers if they don't exist."""
        directory = os.path.dirname(self.file_path)
        if directory:
            ensure_directory(directory)

        if not os.path.exists(self.file_path):
            try:
                with open(self.file_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "language", "title", "audio_generated"])
                logger.info(f"Created topics log file: {self.file_path}")
            except OSError as e:
                logger.error(f"Failed to create topics log file: {e}")
                raise

    async def _read_all_rows(self) -> List[Dict[str, str]]:
        """
        Read all rows from the CSV file in a thread-safe manner.

        Uses asyncio.to_thread to avoid blocking the event loop,
        and the lock to prevent reading during a write.

        Returns:
            List of rows as dictionaries. Returns empty list if file missing.
        """
        async with self._lock:
            try:
                # Offload blocking I/O to a thread
                return await asyncio.to_thread(self._read_all_rows_sync)
            except Exception as e:
                logger.error(f"Error reading topics log: {e}")
                return []

    def _read_all_rows_sync(self) -> List[Dict[str, str]]:
        """Synchronous file read. Called via asyncio.to_thread."""
        try:
            with open(self.file_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except FileNotFoundError:
            logger.warning("Topics log file not found, returning empty list.")
            return []
        except csv.Error as e:
            logger.error(f"CSV parsing error: {e}")
            return []

    async def append_topic(
        self, language: str, title: str, audio_generated: str = "false"
    ) -> None:
        """
        Append a new publication record to the log.

        Args:
            language: Language code (e.g., "ar").
            title: The published topic title.
            audio_generated: One of "true", "false", "failed".
        """
        # Validate inputs
        if not language or not title:
            logger.warning("Attempted to log topic with empty language or title, skipping.")
            return

        valid_audio = {"true", "false", "failed"}
        if audio_generated not in valid_audio:
            logger.warning(
                f"Invalid audio_generated value '{audio_generated}', defaulting to 'false'"
            )
            audio_generated = "false"

        timestamp = utc_now_iso()

        async with self._lock:
            try:
                # Offload blocking I/O to a thread
                await asyncio.to_thread(
                    self._append_topic_sync, timestamp, language, title, audio_generated
                )
                logger.debug(f"Logged topic: [{language}] {title} (audio={audio_generated})")
            except OSError as e:
                logger.error(f"Failed to append to topics log: {e}")
                # Non-fatal; publication can still succeed

    def _append_topic_sync(
        self, timestamp: str, language: str, title: str, audio_generated: str
    ) -> None:
        """Synchronous append operation."""
        with open(self.file_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, language, title, audio_generated])

    async def get_recent_titles(self, language: str, limit: int = 20) -> List[str]:
        """
        Retrieve the most recent published titles for a language.

        Args:
            language: Language code filter.
            limit: Maximum number of titles to return.

        Returns:
            List of titles, most recent first.
        """
        rows = await self._read_all_rows()
        # Filter and sort by timestamp descending
        lang_rows = [r for r in rows if r.get("language") == language]
        lang_rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        titles = [r["title"] for r in lang_rows[:limit] if r.get("title")]
        return titles

    async def get_today_count(self, language: Optional[str] = None) -> int:
        """
        Count publications that occurred today (UTC date).

        Args:
            language: Optional language filter.

        Returns:
            Number of publications today.
        """
        today_iso = utc_now_iso()[:10]  # YYYY-MM-DD
        rows = await self._read_all_rows()
        count = 0
        for row in rows:
            timestamp = row.get("timestamp", "")
            if not timestamp.startswith(today_iso):
                continue
            if language is not None and row.get("language") != language:
                continue
            count += 1
        return count

    # Alias for compatibility with existing handler code
    async def get_today_topics_count(self, language: str) -> int:
        """
        Count today's publications for a specific language.
        (Alias for get_today_count with language parameter)
        """
        return await self.get_today_count(language)

    async def get_stats(self) -> Dict[str, Any]:
        """
        Retrieve overall publication statistics.

        Returns:
            Dictionary with total counts, today counts, per-language breakdown,
            and audio generation statistics.
        """
        rows = await self._read_all_rows()
        today_iso = utc_now_iso()[:10]

        total = len(rows)
        today_total = 0
        by_language: Dict[str, int] = {}
        audio_stats: Dict[str, int] = {"true": 0, "false": 0, "failed": 0}

        for row in rows:
            lang = row.get("language", "unknown")
            by_language[lang] = by_language.get(lang, 0) + 1

            audio = row.get("audio_generated", "false")
            if audio in audio_stats:
                audio_stats[audio] += 1

            timestamp = row.get("timestamp", "")
            if timestamp.startswith(today_iso):
                today_total += 1

        return {
            "total_publications": total,
            "publications_today": today_total,
            "by_language": by_language,
            "audio_stats": audio_stats,
        }

    async def get_recent_publications(self, limit: int = 10) -> List[Dict[str, str]]:
        """
        Return the most recent publication records for admin display.

        Args:
            limit: Number of records to return.

        Returns:
            List of row dictionaries sorted by timestamp descending.
        """
        rows = await self._read_all_rows()
        rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return rows[:limit]