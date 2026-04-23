"""
Image generation module for Smart Content Bot.

Creates article and quote images using Pillow, with multilingual support
including RTL text reshaping for Arabic and other languages.

Uses asyncio.to_thread to offload CPU-intensive image operations.
"""

import asyncio
import os
import random
import time
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from src.utils.helpers import ensure_directory, generate_unique_filename
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Conditional imports for Arabic reshaping (optional but recommended)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False
    logger.warning(
        "arabic_reshaper or python-bidi not installed. "
        "Arabic text may not render correctly."
    )

# Set of language codes that require RTL text processing
RTL_LANGUAGES = {"ar", "fa", "he", "ur"}

# Default fallback font to use when no TTF is available
# This is a basic bitmap font that comes with Pillow
DEFAULT_BITMAP_FONT = ImageFont.load_default()


class ImageGeneratorError(Exception):
    """Raised when image generation fails irrecoverably."""
    pass


class ImageGenerator:
    """
    Generates visually appealing images for article titles and quotes.
    Supports multiple languages, RTL text, and custom backgrounds/fonts.
    """

    # Default text area dimensions (relative to image size)
    TEXT_AREA_PADDING = 40
    TITLE_MAX_WIDTH_RATIO = 0.8
    TITLE_MAX_HEIGHT_RATIO = 0.3

    QUOTE_MAX_WIDTH_RATIO = 0.75
    QUOTE_MAX_HEIGHT_RATIO = 0.6

    def __init__(
        self,
        backgrounds_dir: str = "assets/backgrounds",
        fonts_dir: str = "assets/fonts",
        output_dir: str = "data/media/images",
        image_size: Tuple[int, int] = (1024, 1024),
    ) -> None:
        """
        Initialize the image generator.

        Args:
            backgrounds_dir: Directory containing background images.
            fonts_dir: Directory with language-specific font subdirectories.
            output_dir: Where generated images will be saved.
            image_size: Width and height of output images in pixels.
        """
        self.backgrounds_dir = backgrounds_dir
        self.fonts_dir = fonts_dir
        self.output_dir = output_dir
        self.image_size = image_size

        # Scan backgrounds
        self.background_paths = self._scan_backgrounds()
        logger.info(f"Found {len(self.background_paths)} background images")

        # Build font map
        self.font_map = self._build_font_map()
        logger.info(f"Loaded fonts for languages: {list(self.font_map.keys())}")

        # Ensure output directory exists
        ensure_directory(self.output_dir)

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

    def _scan_backgrounds(self) -> List[str]:
        """Return list of background image paths."""
        if not os.path.isdir(self.backgrounds_dir):
            logger.warning(f"Backgrounds directory not found: {self.backgrounds_dir}")
            return []
        files = []
        for f in os.listdir(self.backgrounds_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(self.backgrounds_dir, f))
        return files

    def _build_font_map(self) -> dict:
        """
        Build mapping from language code to font file path.
        Expects structure: fonts_dir/{language_code}/somefont.ttf
        """
        font_map = {}
        if not os.path.isdir(self.fonts_dir):
            logger.warning(f"Fonts directory not found: {self.fonts_dir}")
            return font_map

        for lang in os.listdir(self.fonts_dir):
            lang_path = os.path.join(self.fonts_dir, lang)
            if os.path.isdir(lang_path):
                for f in os.listdir(lang_path):
                    if f.lower().endswith((".ttf", ".otf")):
                        font_map[lang] = os.path.join(lang_path, f)
                        break  # Use first found font
        return font_map

    def _get_background_path(self) -> Optional[str]:
        """Select random background image or None if none available."""
        if self.background_paths:
            return random.choice(self.background_paths)
        return None

    def _get_font_path(self, language: str) -> str:
        """
        Get font path for language, falling back to default if needed.
        """
        # Direct match
        if language in self.font_map:
            return self.font_map[language]

        # Try to find any font in language subdirectory
        lang_dir = os.path.join(self.fonts_dir, language)
        if os.path.isdir(lang_dir):
            for f in os.listdir(lang_dir):
                if f.lower().endswith((".ttf", ".otf")):
                    return os.path.join(lang_dir, f)

        # Fallback to default
        default_dir = os.path.join(self.fonts_dir, "default")
        if os.path.isdir(default_dir):
            for f in os.listdir(default_dir):
                if f.lower().endswith((".ttf", ".otf")):
                    return os.path.join(default_dir, f)

        # Absolute fallback – will raise error later but better than returning a non-existent path
        logger.warning(f"No TTF/OTF font found for language '{language}'.")
        return ""

    def _is_rtl_language(self, language: str) -> bool:
        """Check if a language code typically uses right-to-left script."""
        return language in RTL_LANGUAGES

    def _prepare_text(self, text: str, language: str) -> str:
        """
        Reshape and apply bidi for RTL languages if needed.
        """
        if self._is_rtl_language(language) and ARABIC_SUPPORT:
            try:
                reshaped = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped)
                return bidi_text
            except Exception as e:
                logger.error(f"Arabic reshaping failed: {e}")
                return text
        return text

    def _load_font_safe(self, font_path: str, size: int) -> ImageFont.FreeTypeFont:
        """
        Load a TrueType font, falling back to default bitmap font on error.
        Returns an object that supports at least the methods we need.
        """
        if font_path and os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except OSError as e:
                logger.warning(f"Could not load font {font_path}: {e}. Using fallback.")

        # Fallback to default bitmap font (may not support size or getbbox well)
        # For our purposes, we'll try to use a built-in fixed font
        return ImageFont.load_default()

    def _get_text_dimensions(
        self, text: str, font: ImageFont.FreeTypeFont
    ) -> Tuple[int, int]:
        """
        Get width and height of a text string using font.getbbox().
        Falls back to legacy method for older Pillow versions.
        """
        try:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for default bitmap font (deprecated but still works)
            width, height = font.getsize(text)
            return width, height

    def _wrap_text(
        self, text: str, font: ImageFont.FreeTypeFont, max_width: int
    ) -> List[str]:
        """
        Wrap text to fit within max_width pixels.
        Handles both space-separated languages and character-based for RTL.
        """
        # If the text contains spaces, treat as word-based
        if " " in text:
            return self._wrap_text_by_words(text, font, max_width)
        else:
            # For languages without spaces (e.g., Chinese, Japanese) or long RTL strings
            # Use character-based wrapping
            return self._wrap_text_by_characters(text, font, max_width)

    def _wrap_text_by_words(
        self, text: str, font: ImageFont.FreeTypeFont, max_width: int
    ) -> List[str]:
        """Wrap text at word boundaries."""
        words = text.split()
        lines = []
        current_line = []
        current_width = 0

        space_width, _ = self._get_text_dimensions(" ", font)

        for word in words:
            word_width, _ = self._get_text_dimensions(word, font)
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width + space_width
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width + space_width

        if current_line:
            lines.append(" ".join(current_line))

        # If no lines formed (e.g., single word wider than max_width)
        if not lines and text:
            lines = self._wrap_text_by_characters(text, font, max_width)

        return lines

    def _wrap_text_by_characters(
        self, text: str, font: ImageFont.FreeTypeFont, max_width: int
    ) -> List[str]:
        """Wrap text character by character (fallback for long words)."""
        lines = []
        line = ""
        line_width = 0

        for char in text:
            char_width, _ = self._get_text_dimensions(char, font)
            if line_width + char_width <= max_width:
                line += char
                line_width += char_width
            else:
                if line:
                    lines.append(line)
                line = char
                line_width = char_width

        if line:
            lines.append(line)

        return lines

    def _calculate_font_size(
        self,
        text: str,
        font_path: str,
        max_width: int,
        max_height: int,
        min_size: int = 20,
        max_size: int = 120,
    ) -> int:
        """
        Binary search to find largest font size that fits within bounding box.
        """
        best_size = min_size
        low, high = min_size, max_size

        while low <= high:
            mid = (low + high) // 2
            font = self._load_font_safe(font_path, mid)
            lines = self._wrap_text(text, font, max_width)

            # Calculate total height
            total_height = 0
            for line in lines:
                _, line_height = self._get_text_dimensions(line, font)
                total_height += line_height
            total_height += (len(lines) - 1) * 5  # line spacing

            if total_height <= max_height:
                best_size = mid
                low = mid + 1
            else:
                high = mid - 1

        return best_size

    def _create_base_image(self) -> Image.Image:
        """
        Create a base image: either load background or generate gradient.
        """
        bg_path = self._get_background_path()
        if bg_path:
            try:
                img = Image.open(bg_path).convert("RGB")
                return img.resize(self.image_size, Image.Resampling.LANCZOS)
            except Exception as e:
                logger.error(f"Failed to load background {bg_path}: {e}")

        # Fallback: generate a pleasing vertical gradient
        logger.debug("Generating gradient background")
        img = Image.new("RGB", self.image_size, color=(30, 30, 40))
        draw = ImageDraw.Draw(img)
        width, height = self.image_size

        # Draw gradient using small rectangles (much faster than pixel loop)
        for y in range(0, height, 4):
            # Linear interpolation between two colors
            ratio = y / height
            r = int(30 + 30 * ratio)
            g = int(30 + 40 * ratio)
            b = int(60 + 40 * ratio)
            draw.rectangle([(0, y), (width, y + 4)], fill=(r, g, b))

        return img

    def _draw_text_with_alignment(
        self,
        draw: ImageDraw.Draw,
        lines: List[str],
        font: ImageFont.FreeTypeFont,
        area: Tuple[int, int, int, int],  # (x, y, width, height)
        language: str,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        shadow: bool = True,
    ) -> None:
        """
        Draw wrapped text with proper alignment (RTL right-aligned, LTR centered).
        """
        x, y, width, height = area
        line_spacing = 5

        # Calculate total text height
        total_text_height = 0
        line_heights = []
        for line in lines:
            _, h = self._get_text_dimensions(line, font)
            line_heights.append(h)
            total_text_height += h
        total_text_height += (len(lines) - 1) * line_spacing

        current_y = y + (height - total_text_height) // 2

        for i, line in enumerate(lines):
            line_width, line_height = self._get_text_dimensions(line, font)

            # Horizontal alignment
            if self._is_rtl_language(language):
                # Right alignment for RTL
                line_x = x + width - line_width - self.TEXT_AREA_PADDING
            else:
                # Centered for LTR
                line_x = x + (width - line_width) // 2

            # Shadow effect (offset black text behind)
            if shadow:
                draw.text(
                    (line_x + 2, current_y + 2),
                    line,
                    font=font,
                    fill=(0, 0, 0),  # opaque black for shadow
                )

            draw.text((line_x, current_y), line, font=font, fill=text_color)

            current_y += line_heights[i] + line_spacing

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    async def generate_article_image(
        self, title: str, language: str, category: Optional[str] = None
    ) -> str:
        """
        Generate an image for the article title.

        Args:
            title: The topic title.
            language: Language code.
            category: Optional category label.

        Returns:
            Path to the generated image file.

        Raises:
            ImageGeneratorError: If image creation fails.
        """
        return await asyncio.to_thread(
            self._generate_article_image_sync, title, language, category
        )

    def _generate_article_image_sync(
        self, title: str, language: str, category: Optional[str]
    ) -> str:
        """Synchronous implementation of article image generation."""
        try:
            img = self._create_base_image()
            draw = ImageDraw.Draw(img)

            # Prepare text
            title = self._prepare_text(title, language)

            # Define text area for title
            title_width = int(self.image_size[0] * self.TITLE_MAX_WIDTH_RATIO)
            title_height = int(self.image_size[1] * self.TITLE_MAX_HEIGHT_RATIO)
            title_x = (self.image_size[0] - title_width) // 2
            title_y = int(self.image_size[1] * 0.35)

            # Get font and size
            font_path = self._get_font_path(language)
            font_size = self._calculate_font_size(
                title, font_path, title_width, title_height
            )
            font = self._load_font_safe(font_path, font_size)

            lines = self._wrap_text(title, font, title_width)

            # Draw category if provided
            if category:
                category = self._prepare_text(category, language)
                cat_font_size = max(24, font_size // 2)
                cat_font = self._load_font_safe(font_path, cat_font_size)
                cat_width, _ = self._get_text_dimensions(category, cat_font)
                cat_x = (self.image_size[0] - cat_width) // 2
                cat_y = title_y - cat_font_size - 20
                draw.text((cat_x, cat_y), category, font=cat_font, fill=(200, 200, 200))

            # Draw title
            self._draw_text_with_alignment(
                draw,
                lines,
                font,
                (title_x, title_y, title_width, title_height),
                language,
                text_color=(255, 255, 255),
            )

            # Save image
            filename = generate_unique_filename("article", "png")
            output_path = os.path.join(self.output_dir, filename)
            img.save(output_path, "PNG")
            logger.debug(f"Article image saved: {output_path}")

            return output_path

        except Exception as e:
            logger.exception("Failed to generate article image")
            raise ImageGeneratorError(f"Article image generation failed: {e}") from e

    async def generate_quote_image(
        self, quote_text: str, quote_author: str, language: str
    ) -> str:
        """
        Generate an image for the quote.

        Args:
            quote_text: The quote content.
            quote_author: The quote author (or attribution).
            language: Language code.

        Returns:
            Path to the generated image file.

        Raises:
            ImageGeneratorError: If image creation fails.
        """
        return await asyncio.to_thread(
            self._generate_quote_image_sync, quote_text, quote_author, language
        )

    def _generate_quote_image_sync(
        self, quote_text: str, quote_author: str, language: str
    ) -> str:
        """Synchronous implementation of quote image generation."""
        try:
            img = self._create_base_image()
            draw = ImageDraw.Draw(img)

            # Prepare text
            quote_text = self._prepare_text(quote_text, language)
            quote_author = self._prepare_text(quote_author, language)

            # Define text area for quote
            quote_width = int(self.image_size[0] * self.QUOTE_MAX_WIDTH_RATIO)
            quote_height = int(self.image_size[1] * self.QUOTE_MAX_HEIGHT_RATIO)
            quote_x = (self.image_size[0] - quote_width) // 2
            quote_y = int(self.image_size[1] * 0.3)

            # Get font and size for quote text
            font_path = self._get_font_path(language)
            font_size = self._calculate_font_size(
                quote_text, font_path, quote_width, quote_height
            )
            font = self._load_font_safe(font_path, font_size)

            lines = self._wrap_text(quote_text, font, quote_width)

            # Draw quote text
            self._draw_text_with_alignment(
                draw,
                lines,
                font,
                (quote_x, quote_y, quote_width, quote_height),
                language,
                text_color=(255, 255, 200),
            )

            # Draw author
            if quote_author:
                author_font_size = max(20, font_size // 2)
                author_font = self._load_font_safe(font_path, author_font_size)
                author_width, _ = self._get_text_dimensions(quote_author, author_font)
                author_x = (self.image_size[0] - author_width) // 2
                author_y = quote_y + quote_height + 40
                draw.text(
                    (author_x, author_y),
                    quote_author,
                    font=author_font,
                    fill=(200, 200, 200),
                )

            # Save image
            filename = generate_unique_filename("quote", "png")
            output_path = os.path.join(self.output_dir, filename)
            img.save(output_path, "PNG")
            logger.debug(f"Quote image saved: {output_path}")

            return output_path

        except Exception as e:
            logger.exception("Failed to generate quote image")
            raise ImageGeneratorError(f"Quote image generation failed: {e}") from e

    def cleanup_temp_images(self, age_minutes: int = 60) -> int:
        """
        Delete image files older than specified age.

        Args:
            age_minutes: Minimum age in minutes for deletion.

        Returns:
            Number of files deleted.
        """
        if not os.path.isdir(self.output_dir):
            return 0

        now = time.time()
        cutoff = now - (age_minutes * 60)
        deleted = 0

        for filename in os.listdir(self.output_dir):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            filepath = os.path.join(self.output_dir, filename)
            try:
                mtime = os.path.getmtime(filepath)
                if mtime < cutoff:
                    os.remove(filepath)
                    deleted += 1
            except OSError as e:
                logger.warning(f"Failed to delete {filepath}: {e}")

        if deleted:
            logger.info(f"Cleaned up {deleted} temporary image(s)")
        return deleted