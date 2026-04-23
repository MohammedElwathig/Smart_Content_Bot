"""
Audio generation decision helper.

Provides a probability-based function to determine whether to generate
a podcast audio file for a given publication. This allows controlling
API costs and processing time by generating audio only for a fraction
of posts.

Usage:
    from src.tts.audio_decision import should_generate_audio

    if should_generate_audio():
        # Generate and attach podcast audio
        pass
"""

import random
from typing import Optional

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default fallback value if settings provides an invalid denominator
DEFAULT_DENOMINATOR = 4


def should_generate_audio(denominator: Optional[int] = None) -> bool:
    """
    Determine whether audio should be generated based on a 1/N probability.

    The probability is defined as 1 / denominator. For example:
    - denominator = 1 -> always generate (100%)
    - denominator = 2 -> generate half the time (50%)
    - denominator = 4 -> generate quarter of the time (25%)

    Args:
        denominator: Optional override for the probability denominator.
                     If not provided, uses `settings.audio_ratio_denominator`.

    Returns:
        True if audio should be generated, False otherwise.

    Raises:
        ValueError: If denominator is less than 1 (after fallback).
    """
    # Determine the effective denominator
    if denominator is None:
        denominator = settings.audio_ratio_denominator

    # Validate and use fallback if necessary
    if denominator <= 0:
        logger.warning(
            f"Invalid audio ratio denominator: {denominator}. "
            f"Falling back to default value {DEFAULT_DENOMINATOR}."
        )
        denominator = DEFAULT_DENOMINATOR

    # Generate a random integer between 1 and denominator inclusive.
    # If it equals 1, the event occurs.
    result = random.randint(1, denominator) == 1

    # Log at debug level since this is a frequent, low-importance decision
    logger.debug(
        f"Audio generation decision: {result} (probability 1/{denominator} = {100/denominator:.1f}%)"
    )

    return result


# -----------------------------------------------------------------------------
# Utility function for logging the decision rate over time
# -----------------------------------------------------------------------------

def get_probability_percentage(denominator: Optional[int] = None) -> float:
    """
    Return the probability of audio generation as a percentage.

    Args:
        denominator: Optional override for the denominator.

    Returns:
        Percentage value between 0.0 and 100.0.
    """
    if denominator is None:
        denominator = settings.audio_ratio_denominator
    if denominator <= 0:
        denominator = DEFAULT_DENOMINATOR
    return 100.0 / denominator