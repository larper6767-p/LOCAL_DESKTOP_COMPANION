# MARIN-LIVE — Friend Test Build

A desktop AI companion with voice input, speech-to-text, an LLM brain, text-to-speech, and an animated desktop pet.

This is a **friend-test** build: I've stripped out my personal data (my name, my memory, my voice profile, window position, and any messages). You set it up with **your own** API keys.


##  One-time install (takes a few minutes)

1. **Install Python** 3.11–3.14 from https://www.python.org/downloads/ — tick **"Add Python to PATH"** during install.
2. Double-click **`setup.bat`**. It creates a private `.venv` and installs everything from `requirements.lock` (exact tested versions).
3. Double-click **`run.bat`** to start the app. On first launch you'll be asked for your two API keys.

That's it. After setup, you just use `run.bat`.

> No model files are included in this build (I'm checking the model's license first), so the pet appears as an animated **sprite** instead of the 3D Live2D model. That's expected and fine.


##  Getting your keys

The app needs two free-to-sign-up services:

| Service | What it does | Where to get a key |

| **Fish Audio** | The voice (text-to-speech) | https://fish.audio → Account → API Keys |
| **Cerebras** | The "brain" (LLM chat) | https://cloud.cerebras.ai → API Keys |

Put them in the file **`.env`** (copy it from `.env.example`), or just enter them when the app asks on first launch.

>  **Never share your `.env` or your keys.** 

## Setting up a voice for your character 
   
   **From fish audio** go into discovery from https://fish.audio/app/discovery/ look through your favourite voice and copy its voice id and paste it into your config 


##  What data goes where

- **Text you say / type** is sent to **Cerebras** (the LLM) to generate her replies.
- Her **spoken reply** is rendered by **Fish Audio** (your text goes to their TTS).
- **All data stays on your own machine** locally: the Whisper speech-to-text runs 100% offline on your own computer — your voice is not uploaded for transcription.

### Where files are stored & how to delete them

Everything the app remembers lives in this folder (the same place as this README):

| Path | Contents |

| `config/memory.json` | Facts she remembers about you |
| `config/relationship.json` | A "closeness" score that grows with use |
| `config/pet_state.json` | Window position / size / settings |
| `config/episodes/` | A dated log of your conversations |
| `logs/` | Run logs (no message text, just activity) |

**To wipe all personal data:** delete the whole app folder. (Or just delete `config/memory.json`, `config/episodes/`, and `config/relationship.json` to reset her memory while keeping the app.)

`diagnose_out.txt` (from §5) contains **no** conversation content — it's just a system-health report.


##  Privacy note about this package

The `.env.example` and built package contain **no real keys**. Anything in `.env` is yours. The developer's real `.env`, voice ID, memory, and messages are **not** shipped with this build.


##  If something isn't working

Double-click **`diagnose.bat`** — it writes a health report to **`diagnose_out.txt`** and prints the same on screen. Send me that file (paste it into a chat). It has no private conversation content.

Common issues:
- **"Python not found"** in setup → you skipped "Add Python to PATH".
- **No voice / no reply** → your Fish Audio key is wrong, or the voice isn't available on your account.
- **Slow first reply** → Whisper downloads its small model on first run (a few MB), then it's cached.
- **App won't start / blank** → run `diagnose.bat` and send me `diagnose_out.txt`.


##  Model / SDK license note

- The animated **Live2D model** is NOT for commercial use and only testing 



