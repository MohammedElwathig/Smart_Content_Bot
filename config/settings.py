"""
Configuration module for Smart Content Bot.
Loads environment variables, validates them, and provides a single settings object.
"""

import os
import sys
from typing import Dict, List

from pydantic import Field, ValidationError, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    All mandatory fields must be present or the application will fail fast.
    """

    # Core settings
    telegram_bot_token: str = Field(..., description="Telegram Bot Token from @BotFather")
    gemini_api_keys_str: str = Field(
        ..., alias="GEMINI_API_KEYS", description="Comma-separated Gemini API keys"
    )
    admin_user_ids_str: str = Field(
        ..., alias="ADMIN_USER_IDS", description="Comma-separated Telegram user IDs"
    )

    # Optional audio settings
    audio_ratio_denominator: int = Field(
        4,
        alias="AUDIO_RATIO_DENOMINATOR",
        description="Probability denominator for podcast generation (1/N chance)",
    )
    tts_rate: str = Field(
        "+0%",
        alias="TTS_RATE",
        description="Speaking rate for Edge-TTS",
    )
    tts_pitch: str = Field(
        "+0Hz",
        alias="TTS_PITCH",
        description="Pitch change for Edge-TTS",
    )

    # Scheduler settings
    publish_interval_minutes: int = Field(
        60,
        alias="PUBLISH_INTERVAL_MINUTES",
        description="Interval in minutes between automatic publications",
    )

    # Logging settings
    log_level: str = Field(
        "INFO",
        alias="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    # Render-specific
    port: int = Field(
        10000,
        alias="PORT",
        description="Port for health check HTTP server (set by Render)",
    )
    environment: str = Field(
        "production",
        alias="ENVIRONMENT",
        description="Deployment environment (development/production)",
    )

    # Pydantic v2 model configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars
        case_sensitive=False,
    )

    # --- Computed Properties (exposed as fields via @computed_field) ---

    @computed_field
    @property
    def gemini_api_keys(self) -> List[str]:
        """Parse comma-separated API keys into a list of non-empty strings."""
        keys = [k.strip() for k in self.gemini_api_keys_str.split(",") if k.strip()]
        if not keys:
            raise ValueError("GEMINI_API_KEYS must contain at least one valid key")
        return keys

    @computed_field
    @property
    def admin_user_ids(self) -> List[int]:
        """Parse comma-separated user IDs into a list of integers."""
        ids = []
        for item in self.admin_user_ids_str.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                ids.append(int(item))
            except ValueError:
                raise ValueError(f"Invalid admin user ID: '{item}' must be an integer")
        if not ids:
            raise ValueError("ADMIN_USER_IDS must contain at least one valid ID")
        return ids

    @computed_field
    @property
    def tts_voice_overrides(self) -> Dict[str, str]:
        """
        Extract dynamic TTS voice overrides from environment variables.
        Example: TTS_VOICE_ar=ar-EG-SalmaNeural -> {"ar": "ar-EG-SalmaNeural"}

        Note: This reads os.environ directly to catch any variable with prefix
        TTS_VOICE_, even those not explicitly defined as fields. This allows
        flexible addition of language-specific voices without code changes.
        """
        overrides = {}
        prefix = "TTS_VOICE_"
        for key, value in os.environ.items():
            if key.startswith(prefix) and len(key) > len(prefix):
                lang_code = key[len(prefix):].lower()
                overrides[lang_code] = value
        return overrides

    # --- Validators ---

    @field_validator("audio_ratio_denominator")
    @classmethod
    def validate_audio_ratio(cls, v: int) -> int:
        """Ensure denominator is positive."""
        if v <= 0:
            raise ValueError("AUDIO_RATIO_DENOMINATOR must be a positive integer")
        return v

    @field_validator("publish_interval_minutes")
    @classmethod
    def validate_interval(cls, v: int) -> int:
        """Ensure interval is at least 1 minute."""
        if v < 1:
            raise ValueError("PUBLISH_INTERVAL_MINUTES must be at least 1")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Normalize log level to uppercase and validate."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return upper

    # --- Utility Methods ---

    def redacted_dict(self) -> Dict[str, str]:
        """
        Return a dictionary representation with sensitive values redacted.
        Useful for logging configuration at startup.
        """
        d = self.model_dump()
        d["telegram_bot_token"] = "[REDACTED]"
        d["gemini_api_keys_str"] = "[REDACTED]"
        # Also redact the computed list property if present
        if "gemini_api_keys" in d:
            d["gemini_api_keys"] = ["[REDACTED]"] * len(self.gemini_api_keys)
        return d


# Create a global settings instance. This will validate on import.
try:
    settings = Settings()
except ValidationError as e:
    print("ERROR: Invalid configuration. Please check your environment variables.")
    print(e)
    sys.exit(1)
except ValueError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# For convenience, log configuration on startup (will be done in main.py)