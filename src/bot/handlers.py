"""
Command handlers for Smart Content Bot.

Defines admin-only commands for monitoring and controlling the bot:
/start, /help, /status, /cache, /force, /refresh_cache
"""

from typing import Any, Dict

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from config.settings import settings
from src.storage.languages import reload_languages, get_languages
from src.utils.logger import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------------------
# Admin Authorization
# -------------------------------------------------------------------------

def is_admin(user_id: int) -> bool:
    """Check if user ID is in the authorized admins list."""
    return user_id in settings.admin_user_ids


def admin_only(func):
    """
    Decorator to restrict command access to authorized admins.
    Sends "Access denied" message to unauthorized users.
    """
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user or not is_admin(user.id):
            await update.message.reply_text("⛔ Access denied. Admin only.")
            logger.warning(f"Unauthorized access attempt from user {user.id if user else 'unknown'}")
            return
        return await func(update, context)
    return wrapper


# -------------------------------------------------------------------------
# Helper to format status messages
# -------------------------------------------------------------------------

def _format_status_html(
    key_status: Dict[str, Any],
    pub_stats: Dict[str, Any],
    cache_status: Dict[str, Any],
    next_run: str = "N/A",
) -> str:
    """Format a comprehensive status message in HTML."""
    total_keys = key_status.get('total_keys', 0)
    active_keys = key_status.get('active_keys', 0)
    exhausted_keys = key_status.get('exhausted_keys', 0)

    lines = [
        "<b>📊 Smart Content Bot Status</b>",
        "",
        "<b>🔑 Gemini API Keys:</b>",
        f"  • Total: {total_keys}",
        f"  • Active: {active_keys}",
        f"  • Exhausted: {exhausted_keys}",
    ]

    # Add warning if low on active keys
    if active_keys == 0:
        lines.append("  ⚠️ <b>WARNING: No active API keys!</b>")
    elif active_keys < total_keys / 2:
        lines.append("  ⚠️ Some keys are exhausted.")

    lines.extend(["", "<b>📈 Publications:</b>"])

    total_pubs = pub_stats.get('total_publications', 0)
    today_pubs = pub_stats.get('publications_today', 0)
    lines.append(f"  • Total: {total_pubs}")
    lines.append(f"  • Today: {today_pubs}")

    # By language
    by_lang = pub_stats.get('by_language', {})
    if by_lang:
        lang_lines = [f"  • {lang}: {count}" for lang, count in by_lang.items()]
        lines.append("\n".join(lang_lines))

    # Audio stats
    audio = pub_stats.get('audio_stats', {})
    if audio:
        lines.append("")
        lines.append("<b>🎵 Audio Generation:</b>")
        lines.append(f"  • Success: {audio.get('true', 0)}")
        lines.append(f"  • Skipped: {audio.get('false', 0)}")
        lines.append(f"  • Failed: {audio.get('failed', 0)}")

    lines.append("")
    lines.append("<b>📅 Daily Cache:</b>")
    for lang, info in cache_status.items():
        valid = "✅" if info.get('is_valid') else "⚠️"
        remaining = info.get('remaining', 0)
        lines.append(f"  {valid} {lang}: {remaining} topics remaining")

    lines.append("")
    lines.append(f"<b>⏰ Next Publication:</b> {next_run}")

    return "\n".join(lines)


# -------------------------------------------------------------------------
# Command Handlers
# -------------------------------------------------------------------------

@admin_only
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /start and /help - Show welcome message and available commands.
    """
    help_text = """
<b>🤖 Smart Content Bot</b>

This bot automatically publishes AI-generated content to Telegram channels in multiple languages.

<b>📋 Admin Commands:</b>
/start - Show this help message
/status - Show bot status and statistics
/cache - Show daily topics cache status
/force [lang|all] - Force immediate publication
/refresh_cache [lang|all] - Regenerate daily topic cache

<i>Examples:</i>
<code>/force en</code> - Publish to English channel now
<code>/refresh_cache all</code> - Regenerate cache for all languages
"""
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)


@admin_only
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /status - Show comprehensive bot status.
    """
    services = context.application.bot_data.get("services", {})
    key_manager = services.get("key_manager")
    csv_manager = services.get("csv_manager")
    topic_generator = services.get("topic_generator")
    job_scheduler = services.get("job_scheduler")

    if not all([key_manager, csv_manager, topic_generator, job_scheduler]):
        await update.message.reply_text("⚠️ Services not fully initialized.")
        return

    # Gather status data
    key_status = await key_manager.get_status()      # ✅ الآن async
    pub_stats = await csv_manager.get_stats()
    cache_status = topic_generator.get_cache_status()

    # Get next run time from scheduler
    scheduler_status = job_scheduler.get_status()
    next_run_str = scheduler_status.get("next_run_time", "Not scheduled")

    message = _format_status_html(key_status, pub_stats, cache_status, next_run_str)
    await update.message.reply_text(message, parse_mode=ParseMode.HTML)


@admin_only
async def cache_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /cache - Display current daily cache status.
    """
    services = context.application.bot_data.get("services", {})
    topic_generator = services.get("topic_generator")

    if not topic_generator:
        await update.message.reply_text("⚠️ Topic generator not initialized.")
        return

    cache_status = topic_generator.get_cache_status()
    if not cache_status:
        await update.message.reply_text("No cache data available.")
        return

    lines = ["<b>📅 Daily Topics Cache</b>", ""]
    for lang, info in cache_status.items():
        status_icon = "✅" if info.get('is_valid') else "⚠️"
        date_str = info.get('date', 'unknown')
        remaining = info.get('remaining', 0)
        lines.append(f"{status_icon} <b>{lang.upper()}</b>: {remaining} topics (date: {date_str})")

    lines.append("")
    lines.append("Use <code>/refresh_cache [lang]</code> to regenerate.")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


@admin_only
async def force_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /force [lang|all] - Force immediate publication.
    """
    services = context.application.bot_data.get("services", {})
    job_scheduler = services.get("job_scheduler")

    if not job_scheduler:
        await update.message.reply_text("⚠️ Scheduler not initialized.")
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: <code>/force [language_code|all]</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    target = args[0].lower()
    await update.message.reply_text(f"⏳ Forcing publication for: {target}...")

    try:
        language = None if target == "all" else target
        results = await job_scheduler.force_publish(language)

        if results is None:
            await update.message.reply_text("❌ Failed to execute force publish.")
            return

        if language:
            status = results.get(language, "unknown")
            await update.message.reply_text(f"✅ Publication for '{language}': {status}")
        else:
            success_count = sum(1 for v in results.values() if v == "success")
            await update.message.reply_text(
                f"✅ Publication completed: {success_count}/{len(results)} languages succeeded."
            )
    except Exception as e:
        logger.exception("Force publish failed")
        # اقتطاع رسالة الخطأ إذا كانت طويلة جداً
        error_msg = str(e)[:200]
        await update.message.reply_text(f"❌ Failed to force publish: {error_msg}")


@admin_only
async def refresh_cache_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /refresh_cache [lang|all] - Regenerate daily topic cache.
    Also reloads languages from CSV to pick up any changes.
    """
    services = context.application.bot_data.get("services", {})
    topic_generator = services.get("topic_generator")

    if not topic_generator:
        await update.message.reply_text("⚠️ Topic generator not initialized.")
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: <code>/refresh_cache [language_code|all]</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    target = args[0].lower()
    language = None if target == "all" else target

    await update.message.reply_text(f"🔄 Regenerating cache for: {target}...")

    try:
        # Reload languages from CSV first (async)
        await reload_languages()

        # Regenerate cache
        await topic_generator.regenerate_cache(language)

        if language:
            await update.message.reply_text(f"✅ Cache regenerated for '{language}'.")
        else:
            langs = [lang.code for lang in get_languages()]
            await update.message.reply_text(
                f"✅ Cache regenerated for all languages: {', '.join(langs)}."
            )
    except Exception as e:
        logger.exception("Cache regeneration failed")
        error_msg = str(e)[:200]
        await update.message.reply_text(f"❌ Failed to regenerate cache: {error_msg}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Global error handler for uncaught exceptions.
    """
    logger.error("Unhandled exception:", exc_info=context.error)

    if update and isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ An internal error occurred. Check logs for details.",
            )
        except Exception:
            pass


# -------------------------------------------------------------------------
# Registration
# -------------------------------------------------------------------------

def register_handlers(app: Application, services: Dict[str, Any]) -> None:
    """
    Register all command handlers with the Telegram bot application.

    Args:
        app: The Application instance from python-telegram-bot.
        services: Dictionary containing service instances:
            - telegram_bot
            - topic_generator
            - csv_manager
            - key_manager
            - job_scheduler
    """
    app.bot_data["services"] = services

    app.add_handler(CommandHandler(["start", "help"], start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("cache", cache_cmd))
    app.add_handler(CommandHandler("force", force_cmd))
    app.add_handler(CommandHandler("refresh_cache", refresh_cache_cmd))

    app.add_error_handler(error_handler)

    logger.info("Command handlers registered")