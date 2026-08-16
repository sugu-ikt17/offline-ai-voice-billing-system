# Offline AI Voice Billing System

An offline, voice-driven billing system designed to run on a Raspberry Pi Zero 2 W.
The system records spoken orders, transcribes them locally using **Whisper.cpp**,
parses the order, matches it against a shop's menu, and generates a bill — all
without requiring an internet connection.

---

## Status

**Production-ready.** All core modules implemented:

| Module | Status |
|---|---|
| Menu CRUD | ✅ Complete |
| Order Parser | ✅ Complete |
| Menu Matcher | ✅ Complete |
| Bill Generator | ✅ Complete |
| Receipt Preview | ✅ Complete |
| Browser Printing | ✅ Complete |
| **Whisper.cpp STT** | ✅ **Integrated** |

---

## Target Hardware

- Raspberry Pi Zero 2 W (or any Linux machine)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, FastAPI |
| ORM | SQLAlchemy |
| Database | SQLite |
| Frontend | HTML, CSS, Vanilla JavaScript (ES Modules) |
| Speech Engine | **Whisper.cpp** (offline, on-device) |
| Architecture | Clean Architecture |

---

## Project Structure

```
app/
├── core/                      # Configuration, logging, exceptions
│   ├── config.py              # Settings (env vars: WHISPER_BINARY_PATH, WHISPER_MODEL_PATH)
│   ├── exceptions.py
│   └── logging.py
├── domain/                    # Enterprise business rules
│   ├── entities/              # Menu Item, Order, Bill entities
│   └── interfaces/            # Abstract contracts
├── application/               # Application-specific business rules
│   ├── use_cases/             # Orchestration logic
│   └── services/
│       ├── speech_service.py  # ← SpeechService (transcribe_upload + transcribe)
│       └── ...
├── infrastructure/
│   ├── database/              # SQLite models & repositories
│   └── speech_engine/
│       ├── whisper_cpp_engine.py  # ← Runs whisper.cpp via subprocess
│       └── audio_converter.py     # ← Converts webm/ogg/mp3 → 16kHz WAV
└── presentation/
    └── api/v1/
        ├── speech_routes.py   # ← POST /api/v1/speech/transcribe  ✨ NEW
        ├── voice_routes.py    # Legacy endpoint (WAV-only)
        ├── menu_routes.py
        ├── order_routes.py
        └── bill_routes.py

frontend/
├── index.html
├── js/
│   ├── voice.js   # ← MediaRecorder + Whisper upload (no browser SpeechRecognition)
│   ├── api.js     # ← transcribeVoiceWhisper() + other API calls
│   └── ...
models/whisper/    # whisper.cpp binary + model files (gitignored)
scripts/
└── setup_whisper.sh   # ← One-command Whisper.cpp setup
data/              # SQLite DB + temp audio uploads (gitignored)
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/sugu-ikt17/voice-project.git
cd voice-project
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Install ffmpeg (for non-WAV audio conversion)

```bash
sudo apt install ffmpeg       # Ubuntu / Debian / Raspberry Pi OS
# or
brew install ffmpeg           # macOS
```

> **Note:** If ffmpeg is not installed, the API will still accept WAV uploads
> (the browser already converts recordings to WAV automatically).

### 4. Set up Whisper.cpp (one-time)

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

This will:
- Clone and compile `whisper.cpp` (requires `git`, `make`, `g++`)
- Download the **tiny.en** model (~39 MB) from Hugging Face
- Place both under `models/whisper/`

### 5. Start the backend

```bash
uvicorn app.main:app --reload
```

### 6. Open the frontend

```bash
# Serve the frontend separately (avoids browser CORS issues during dev)
cd frontend && python3 -m http.server 3000
```

Then open http://localhost:3000 in your browser.

---

## Configuration

All settings can be overridden via environment variables or a `.env` file
at the project root.

| Variable | Default | Description |
|---|---|---|
| `WHISPER_BINARY_PATH` | `models/whisper/main` | Path to compiled whisper.cpp binary |
| `WHISPER_MODEL_PATH` | `models/whisper/ggml-tiny.en.bin` | Path to GGML model file |
| `WHISPER_MODEL_NAME` | `ggml-tiny.en.bin` | Model filename (informational) |
| `AUDIO_UPLOAD_DIR` | `data/audio_uploads` | Temp directory for uploaded audio |
| `TAX_RATE` | `0.05` | GST/tax rate applied to bills |

Example `.env` file:

```ini
WHISPER_BINARY_PATH=/home/pi/whisper.cpp/main
WHISPER_MODEL_PATH=/home/pi/whisper.cpp/models/ggml-tiny.en.bin
```

---

## API Reference

### Speech Transcription

#### `POST /api/v1/speech/transcribe`

Transcribes uploaded audio using Whisper.cpp (offline).

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `audio` | file | ✅ | Audio file (wav, webm, ogg, mp3) |

**Response — success (200):**
```json
{
  "success": true,
  "transcript": "2 coffee 2 dosa"
}
```

**Response — engine unavailable (503):**
```json
{
  "success": false,
  "message": "Speech engine unavailable"
}
```

**cURL example:**
```bash
curl -X POST http://localhost:8000/api/v1/speech/transcribe \
  -F "audio=@/path/to/recording.wav"
```

---

### Other Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/menu` | List all menu items |
| POST | `/api/v1/menu` | Create a menu item |
| PUT | `/api/v1/menu/{id}` | Update a menu item |
| DELETE | `/api/v1/menu/{id}` | Delete a menu item |
| POST | `/api/v1/orders/process` | Parse text into a structured order + bill |
| GET | `/api/v1/bills` | List bills |
| GET | `/docs` | Interactive Swagger UI |

---

## Graceful Degradation

If Whisper.cpp is not installed (binary or model missing), the application
**does not crash**. Instead:

- `POST /api/v1/speech/transcribe` returns **503 Service Unavailable**
- All other endpoints (Menu, Orders, Bills) continue working normally
- A warning is logged at startup: `Speech engine not ready: ...`

This allows the system to be demonstrated without Whisper.cpp installed.

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## License

TBD



