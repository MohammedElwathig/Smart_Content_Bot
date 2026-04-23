"""
Intelligent Gemini API key manager with 24-hour exhaustion blacklist.

Provides round-robin key rotation and persistent blacklisting of
rate-limited keys across application restarts.
"""

import asyncio
import json
import os
from typing import Any, Dict, List

from src.utils.helpers import ensure_directory, utc_now_timestamp
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Duration (in seconds) a key remains blacklisted after exhaustion
BLACKLIST_DURATION_SECONDS = 24 * 60 * 60  # 24 hours


class AllKeysExhaustedError(Exception):
    """Raised when no non-blacklisted Gemini API keys are available."""

    pass


class GeminiKeyManager:
    """
    Manages a pool of Gemini API keys with smart rotation and persistence.

    Features:
    - Round-robin selection of available keys.
    - Blacklists keys that return 429 errors for 24 hours.
    - Persists blacklist to disk to survive restarts.
    - Async-safe operations using asyncio.Lock.
    """

    def __init__(
        self,
        api_keys: List[str],
        blacklist_file: str = "data/exhausted_keys.json",
    ) -> None:
        """
        Initialize the key manager.

        Args:
            api_keys: List of Gemini API key strings.
            blacklist_file: Path to JSON file for persistent blacklist.
        """
        if not api_keys:
            raise ValueError("At least one Gemini API key must be provided")

        self._keys = api_keys
        self._blacklist_file = blacklist_file
        self._blacklist: Dict[str, float] = {}
        self._current_index = 0
        self._lock = asyncio.Lock()

        # Synchronous initialization allowed in __init__
        self._load_blacklist_sync()
        self._cleanup_expired_sync()

        active = self.active_key_count_sync
        logger.info(
            f"GeminiKeyManager initialized: {len(self._keys)} total keys, "
            f"{active} active, {len(self._blacklist)} blacklisted"
        )

    # -------------------------------------------------------------------------
    # Synchronous Helpers (for initialization and async wrappers)
    # -------------------------------------------------------------------------

    def _load_blacklist_sync(self) -> None:
        """Load blacklist from JSON file synchronously (used during init)."""
        try:
            with open(self._blacklist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self._blacklist = data
                    logger.debug(f"Loaded blacklist with {len(data)} entries")
                else:
                    logger.warning("Blacklist file malformed, starting fresh")
                    self._blacklist = {}
        except FileNotFoundError:
            logger.debug("No existing blacklist file found, starting fresh")
            self._blacklist = {}
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in blacklist file: {e}, starting fresh")
            self._blacklist = {}

    async def _load_blacklist(self) -> None:
        """Load blacklist from JSON file asynchronously."""
        await asyncio.to_thread(self._load_blacklist_sync)

    def _save_blacklist_sync(self) -> None:
        """Persist current blacklist to JSON file synchronously."""
        directory = os.path.dirname(self._blacklist_file)
        if directory:
            ensure_directory(directory)

        try:
            with open(self._blacklist_file, "w", encoding="utf-8") as f:
                json.dump(self._blacklist, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved blacklist with {len(self._blacklist)} entries")
        except OSError as e:
            logger.error(f"Failed to persist blacklist: {e}. Continuing in-memory only.")
            # Non-fatal; blacklist remains in memory

    async def _save_blacklist(self) -> None:
        """Persist blacklist to JSON file asynchronously."""
        await asyncio.to_thread(self._save_blacklist_sync)

    def _cleanup_expired_sync(self) -> int:
        """
        Remove keys from blacklist whose 24-hour period has passed.

        Returns:
            Number of keys removed.
        """
        now = utc_now_timestamp()
        expired_keys = [
            key for key, ts in self._blacklist.items()
            if now - ts >= BLACKLIST_DURATION_SECONDS
        ]
        for key in expired_keys:
            del self._blacklist[key]
        if expired_keys:
            logger.info(f"Removed {len(expired_keys)} expired key(s) from blacklist")
        return len(expired_keys)

    async def _cleanup_expired(self) -> int:
        """Clean up expired blacklist entries asynchronously."""
        return await asyncio.to_thread(self._cleanup_expired_sync)

    # -------------------------------------------------------------------------
    # Public Properties and Methods
    # -------------------------------------------------------------------------

    @property
    def active_key_count_sync(self) -> int:
        """Number of currently usable (non-blacklisted) keys (non-async)."""
        return len([k for k in self._keys if k not in self._blacklist])

    async def get_active_key_count(self) -> int:
        """Return number of active keys asynchronously (with lock)."""
        async with self._lock:
            return self.active_key_count_sync

    async def get_next_key(self) -> str:
        """
        Retrieve the next available (non-blacklisted) API key.

        Rotates through the key pool in round-robin fashion.

        Returns:
            A valid Gemini API key.

        Raises:
            AllKeysExhaustedError: If no non-blacklisted keys are available.
        """
        async with self._lock:
            # Clean up expired entries first
            await self._cleanup_expired()

            if not self._keys:
                raise AllKeysExhaustedError("No API keys configured")

            # Try all keys starting from current index
            for _ in range(len(self._keys)):
                key = self._keys[self._current_index]
                self._current_index = (self._current_index + 1) % len(self._keys)
                if key not in self._blacklist:
                    logger.debug(f"Providing API key: {key[:8]}...")
                    return key

            # No keys available
            raise AllKeysExhaustedError(
                f"All {len(self._keys)} Gemini API keys are currently exhausted"
            )

    async def mark_key_exhausted(self, key: str) -> None:
        """
        Blacklist a key for 24 hours (e.g., after receiving HTTP 429).

        Args:
            key: The API key to mark as exhausted.
        """
        async with self._lock:
            self._blacklist[key] = utc_now_timestamp()
            await self._save_blacklist()
            logger.warning(
                f"Gemini API key {key[:8]}... marked as exhausted for 24 hours. "
                f"({self.active_key_count_sync} keys remain active)"
            )

    async def get_status(self) -> Dict[str, Any]:
        """
        Return a status dictionary for monitoring/admin commands.

        Returns:
            Dictionary with total, active, exhausted key counts and blacklist details.
        """
        async with self._lock:
            await self._cleanup_expired()
            return {
                "total_keys": len(self._keys),
                "active_keys": self.active_key_count_sync,
                "exhausted_keys": len(self._blacklist),
                "blacklist": dict(self._blacklist),  # copy to avoid mutation
            }

    # -------------------------------------------------------------------------
    # Optional: Validation stub (can be extended later)
    # -------------------------------------------------------------------------

    async def validate_keys(self) -> Dict[str, bool]:
        """
        Test each key with a minimal Gemini API call.

        This is a placeholder; actual validation requires a Gemini client.
        Returns a dummy result indicating all keys are assumed valid.

        Returns:
            Dictionary mapping key (redacted) to validation status.
        """
        # In a real implementation, you'd make a lightweight API call here.
        result = {}
        for key in self._keys:
            redacted = f"{key[:8]}...{key[-4:]}"
            result[redacted] = True  # Placeholder
        return result