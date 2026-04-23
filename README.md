

```markdown
# 🤖 Smart Content Bot

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![Render](https://img.shields.io/badge/Render-Deploy-46E3B7?logo=render)](https://render.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?logo=telegram)](https://core.telegram.org/bots)

**Smart Content Bot** is a next‑generation, multi‑language automated posting bot for Telegram. It combines AI‑generated content (Gemini), custom image creation, and optional podcast audio into a single, stable, and easy‑to‑deploy system.

---

## 🌟 Overview

The bot periodically publishes creative content—articles, quotes, custom images, and short podcast episodes—to Telegram channels in multiple languages. It is designed to be:

- **Rock‑solid stable** – with an intelligent Gemini API key rotation system.
- **Resource efficient** – runs smoothly on Render's free tier.
- **Easy to deploy and customize** – a single `.env` file and flexible `assets/` folder structure.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🧠 **AI‑Powered Content** | Generates structured articles and quotes using `gemini-2.0-flash-exp`. |
| 🔑 **Smart API Key Rotation** | Manages a pool of Gemini keys; exhausted keys are blacklisted for 24 hours automatically. |
| 🎙️ **Podcast Audio (TTS)** | Converts topics into short audio episodes using the free Edge‑TTS service. |
| 🌍 **Multi‑Language Support** | Separate channels per language via a simple `languages.csv` file. |
| 🖼️ **Custom Image Generation** | Creates article title cards and quote images using Pillow. |
| 📅 **Scheduled Publishing** | APScheduler runs the publication job at configurable intervals. |
| 🛡️ **Graceful Degradation** | If audio or image generation fails, the bot still sends the text content. |
| 🩺 **Render‑Ready** | Includes a `/ping` health endpoint and works with external keep‑alive services. |

---

## 📋 Prerequisites

- **Python 3.11** or higher
- A **Telegram Bot Token** – obtain from [@BotFather](https://t.me/BotFather)
- One or more **Google Gemini API Keys** – get them from [Google AI Studio](https://aistudio.google.com/)
- (Optional) Background images and fonts for visual customization

---

## ⚙️ Installation (Local Development)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/smart-content-bot.git
cd smart-content-bot

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and edit the environment file
cp .env.example .env
# Fill in your TELEGRAM_BOT_TOKEN, GEMINI_API_KEYS, and ADMIN_USER_IDS
```

---

📁 Configuration

1. Environment Variables (.env)

Variable Required Description
TELEGRAM_BOT_TOKEN ✅ Your bot token from BotFather.
GEMINI_API_KEYS ✅ Comma‑separated Gemini API keys.
ADMIN_USER_IDS ✅ Comma‑separated Telegram user IDs that can use admin commands.
AUDIO_RATIO_DENOMINATOR ❌ Probability denominator for audio generation (default: 4 → 25%).
TTS_RATE ❌ Speaking rate (e.g., "+0%").
TTS_PITCH ❌ Pitch adjustment (e.g., "+0Hz").
TTS_VOICE_xx ❌ Override the default voice for a language (e.g., TTS_VOICE_ar=ar-EG-SalmaNeural).
PUBLISH_INTERVAL_MINUTES ❌ How often to publish (default: 60).
LOG_LEVEL ❌ Logging verbosity (INFO, DEBUG).

2. Languages and Channels (data/languages.csv)

Create a CSV file with the following columns:

```csv
language_code,channel_id,language_name
ar,-1001234567890,Arabic
en,-1009876543210,English
fr,-1001112223334,French
```

· language_code – ISO 639‑1 two‑letter code.
· channel_id – Telegram channel ID (usually a negative number).
· language_name – Display name (used in logs and admin messages).

3. Assets (Images & Fonts)

Place your background images and fonts in the assets/ folder:

```
assets/
├── backgrounds/          # .jpg or .png images (randomly selected)
└── fonts/
    ├── default/          # Fallback .ttf font
    ├── ar/               # Arabic‑specific font(s)
    ├── en/               # English‑specific font(s)
    └── ...
```

---

🚀 Running Locally

```bash
python main.py
```

The bot will start:

· Polling for Telegram commands
· The scheduled publication job
· A health check HTTP server on port 8080 (or the PORT env variable)

---

☁️ Deployment on Render

This project is optimized for Render's Web Service type (free tier).

Step‑by‑Step

1. Push the code to a GitHub repository.
2. On Render, click New + → Web Service and connect your repo.
3. Render will automatically detect the render.yaml file. If not, configure manually:
   · Runtime: Python 3
   · Build Command: pip install -r requirements.txt
   · Start Command: python main.py
4. Add the required Environment Variables (see table above) in the Render dashboard.
5. Set the Health Check Path to /ping.
6. Click Deploy Web Service.

⏰ Keeping the Bot Awake

Render's free web services sleep after 15 minutes of inactivity. To keep the bot online 24/7:

· Use a free monitoring service like Uptime Robot or cron-job.org.
· Point it to your service URL: https://your-app-name.onrender.com/ping
· Set the check interval to 5 minutes.

This simple external ping will prevent the service from sleeping.

---

🤖 Admin Commands

The bot responds to the following commands (only from users listed in ADMIN_USER_IDS):

Command Description
/start Welcome message and command overview.
/status Shows API key status, publication counts, cache info, and next run time.
/cache Displays the remaining daily topics for each language.
/force [lang] Triggers an immediate publication for a specific language (/force en) or all languages (/force all).
/refresh_cache [lang] Regenerates the daily topic cache for the given language or all languages.

---

🎨 Customization

Adding a New Language

1. Add a new row to data/languages.csv with the language code and channel ID.
2. (Optional) Add a language‑specific font to assets/fonts/<lang>/.
3. (Optional) Set a custom TTS voice via TTS_VOICE_xx in .env.
4. Restart the bot.

Changing the Publication Schedule

Set PUBLISH_INTERVAL_MINUTES in your .env file (e.g., 120 for every 2 hours).

Customizing Image Appearance

· Replace the files in assets/backgrounds/ with your own images.
· Add or replace .ttf fonts in assets/fonts/ to change the typography.

Changing TTS Voices

The default voice map is built into src/tts/edge_tts_service.py. To override a voice, add an environment variable like TTS_VOICE_en=en-GB-SoniaNeural.

---

❗ Troubleshooting

Issue Likely Cause Solution
429 Resource Exhausted Gemini API quota exceeded. Add more keys to GEMINI_API_KEYS. The key rotator will automatically blacklist exhausted keys for 24 hours.
409 Conflict Another instance of the bot is running. Ensure only one Render service is running (Web Service type, not multiple workers).
Bot stops after 15 minutes Render free tier sleep. Set up Uptime Robot to ping the /ping endpoint every 5 minutes.
Health check fails Port not bound correctly or /ping not responding. Verify PORT environment variable and check logs for errors in health_server.py.
Arabic text appears disconnected Missing arabic-reshaper or python-bidi. Ensure both libraries are installed (pip install -r requirements.txt).

---

📁 Project Structure (Simplified)

```
smart_content_bot/
├── main.py                 # Application entry point
├── config/                 # Settings and environment loading
├── src/
│   ├── ai/                 # Gemini client, key manager, schemas, topic generator
│   ├── tts/                # Edge‑TTS service and audio decision
│   ├── image/              # Pillow image generator
│   ├── bot/                # Telegram bot and command handlers
│   ├── scheduler/          # APScheduler job scheduler
│   ├── storage/            # CSV manager and language loader
│   ├── web/                # Health check HTTP server
│   └── utils/              # Logging and helper functions
├── data/                   # CSV logs, language config, exhausted keys cache
├── assets/                 # Backgrounds and fonts
├── logs/                   # Bot runtime logs
├── .env.example            # Template for environment variables
├── requirements.txt        # Python dependencies
└── render.yaml             # Render deployment configuration
```

---

📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

🙏 Acknowledgements

· Google Generative AI for the Gemini API
· Edge‑TTS for free, high‑quality text‑to‑speech
· python-telegram-bot for the excellent Telegram framework
· Render for the generous free tier

```