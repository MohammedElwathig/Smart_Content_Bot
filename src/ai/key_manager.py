"""
Intelligent Gemini API key manager with 24-hour exhaustion blacklist.

Provides round-robin key rotation and persistent blacklisting of
rate-limited keys across application restarts.

Enhanced with:
- Guaranteed lock release to prevent deadlocks.
- Timeout-protected file I/O operations.
- Jitter in blacklist timestamps to avoid thundering herd on re-enable.
- Robust JSON handling for corrupted blacklist files.
"""

import asyncio
import json
import os
import random
from typing import Any, Dict, List, Optional

from src.utils.helpers import ensure_directory, utc_now_timestamp
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Duration (in seconds) a key remains blacklisted after exhaustion
BLACKLIST_DURATION_SECONDS = 24 * 60 * 60  # 24 hours
# Maximum time to wait for file I/O before giving up
FILE_IO_TIMEOUT = 10.0  # seconds


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
    - Async-safe operations using asyncio.Lock with guaranteed release.
    """

    def __init__(
        self,
        api_keys: List[str],
        blacklist_file: str = "data/exhausted_keys.json",
    ) -> None:
        """
        Initialize the key manager.

        Args:
            api_keys: List of Gemini API key strings. Must not be empty.
            blacklist_file: Path to JSON file for persistent blacklist.

        Raises:
            ValueError: If api_keys is empty.
        """
        if not api_keys:
            raise ValueError("At least one Gemini API key must be provided")

        self._keys = api_keys
        self._blacklist_file = blacklist_file
        self._blacklist: Dict[str, float] = {}
        self._current_index = 0
        self._lock = asyncio.Lock()

        # Synchronous initialization is allowed during __init__
        self._load_blacklist_sync()
        self._cleanup_expired_sync()

        active = self.active_key_count_sync
        logger.info(
            f"GeminiKeyManager initialized: {len(self._keys)} total keys, "
            f"{active} active, {len(self._blacklist)} blacklisted"
        )

    # =========================================================================
    # Synchronous Helpers (for initialization and thread-pool offloading)
    # =========================================================================

    def _load_blacklist_sync(self) -> None:
        """
        Load blacklist from JSON file synchronously.

        Handles missing files, malformed JSON, and unexpected data types
        without crashing.
        """
        try:
            with open(self._blacklist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.debug("No existing blacklist file found, starting fresh")
            self._blacklist = {}
            return
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in blacklist file: {e}. Starting fresh.")
            self._blacklist = {}
            return
        except OSError as e:
            logger.error(f"Could not read blacklist file: {e}. Starting fresh.")
            self._blacklist = {}
            return

        # Validate the loaded data is a proper dictionary
        if not isinstance(data, dict):
            logger.warning("Blacklist file does not contain a JSON object. Starting fresh.")
            self._blacklist = {}
            return

        # Filter out any entries with invalid timestamps
        cleaned = {}
        for key, ts in data.items():
            if not isinstance(key, str) or not isinstance(ts, (int, float)):
                logger.debug(f"Skipping invalid blacklist entry: {key!r}")
                continue
            cleaned[key] = float(ts)

        self._blacklist = cleaned
        logger.debug(f"Loaded blacklist with {len(self._blacklist)} valid entries")

    async def _load_blacklist(self) -> None:
        """Load blacklist from JSON file asynchronously, with timeout."""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._load_blacklist_sync),
                timeout=FILE_IO_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("Timeout while loading blacklist file. Proceeding with empty blacklist.")
            self._blacklist = {}

    def _save_blacklist_sync(self) -> bool:
        """
        Persist current blacklist to JSON file synchronously.

        Returns:
            True if saved successfully, False otherwise.
        """
        directory = os.path.dirname(self._blacklist_file)
        if directory:
            try:
                ensure_directory(directory)
            except OSError as e:
                logger.error(f"Could not create directory for blacklist: {e}")
                return False

        try:
            with open(self._blacklist_file, "w", encoding="utf-8") as f:
                json.dump(self._blacklist, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved blacklist with {len(self._blacklist)} entries")
            return True
        except OSError as e:
            logger.error(f"Failed to persist blacklist: {e}. Will remain in memory only.")
            return False

    async def _save_blacklist(self) -> None:
        """Persist blacklist to JSON file asynchronously, with timeout."""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._save_blacklist_sync),
                timeout=FILE_IO_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("Timeout while saving blacklist file. Data remains in memory.")

    def _cleanup_expired_sync(self) -> int:
        """
        Remove keys from blacklist whose 24-hour period has passed.

        Adds a small random jitter (0-60 seconds) to the expiry check
        to prevent multiple keys from being re-enabled simultaneously.

        Returns:
            Number of keys removed.
        """
        now = utc_now_timestamp()
        # Add jitter to avoid thundering herd when multiple keys expire at once
        expiry_threshold = BLACKLIST_DURATION_SECONDS + random.uniform(0, 60)

        expired_keys = [
            key for key, ts in self._blacklist.items()
            if now - ts >= expiry_threshold
        ]
        for key in expired_keys:
            del self._blacklist[key]

        if expired_keys:
            logger.info(f"Removed {len(expired_keys)} expired key(s) from blacklist")
        return len(expired_keys)

    async def _cleanup_expired(self) -> int:
        """Clean up expired blacklist entries asynchronously."""
        # This is fast enough to run inline without thread offloading
        return self._cleanup_expired_sync()

    # =========================================================================
    # Public Properties and Methods
    # =========================================================================

    @property
    def active_key_count_sync(self) -> int:
        """Number of currently usable (non-blacklisted) keys (non-async, no lock)."""
        return sum(1 for k in self._keys if k not in self._blacklist)

    async def get_active_key_count(self) -> int:
        """Return number of active keys asynchronously (with lock)."""
        async with self._lock:
            return self.active_key_count_sync

    async def get_next_key(self) -> str:
        """
        Retrieve the next available (non-blacklisted) API key.

        Rotates through the key pool in round-robin fashion.
        Guaranteed to release the lock even on unexpected errors.

        Returns:
            A valid Gemini API key.

        Raises:
            AllKeysExhaustedError: If no non-blacklisted keys are available.
        """
        if not self._keys:
            raise AllKeysExhaustedError("No API keys configured")

        async with self._lock:
            # Clean up expired entries first
            self._cleanup_expired_sync()

            # Build a set of blacklisted keys for O(1) lookup
            blacklisted = set(self._blacklist.keys())
            available = [k for k in self._keys if k not in blacklisted]

            if not available:
                raise AllKeysExhaustedError(
                    f"All {len(self._keys)} Gemini API keys are currently exhausted. "
                    f"Retry after some keys are re-enabled."
                )

            # Find a key using round-robin (avoid infinite loop by iterating
            # over available keys only, not all keys)
            start_index = self._current_index % len(self._keys)
            for offset in range(len(self._keys)):
                idx = (start_index + offset) % len(self._keys)
                candidate = self._keys[idx]
                if candidate in available:
                    self._current_index = (idx + 1) % len(self._keys)
                    logger.debug(f"Providing API key: {candidate[:8]}...")
                    return candidate

            # Should never reach here if logic is correct, but safety net
            raise AllKeysExhaustedError("Unexpected: no available key found")

    async def mark_key_exhausted(self, key: str) -> None:
        """
        Blacklist a key for 24 hours (e.g., after receiving HTTP 429).

        The lock is always released, even if saving the blacklist fails.

        Args:
            key: The API key to mark as exhausted.
        """
        async with self._lock:
            self._blacklist[key] = utc_now_timestamp()
            remaining = self.active_key_count_sync
            logger.warning(
                f"Gemini API key {key[:8]}... marked as exhausted for 24 hours. "
                f"({remaining} keys remain active)"
            )

        # Save outside the lock to avoid blocking other operations
        await self._save_blacklist()

    async def get_status(self) -> Dict[str, Any]:
        """
        Return a status dictionary for monitoring/admin commands.

        Clean up expired keys before counting to return accurate numbers.

        Returns:
            Dictionary with total, active, exhausted key counts and blacklist details.
        """
        async with self._lock:
            self._cleanup_expired_sync()
            return {
                "total_keys": len(self._keys),
                "active_keys": self.active_key_count_sync,
                "exhausted_keys": len(self._blacklist),
                "blacklist": dict(self._blacklist),  # shallow copy is safe
            }

    # =========================================================================
    # Optional: Validation stub (can be extended with real API calls)
    # =========================================================================

    async def validate_keys(self) -> Dict[str, bool]:
        """
        Test each key with a minimal Gemini API call.

        This is a placeholder; actual validation requires a Gemini client.
        Override or extend this method when a client is available.

        Returns:
            Dictionary mapping redacted key to assumed validity (True).
        """
        return {
            f"{k[:8]}...{k[-4:]}": True
            for k in self._keys
        }