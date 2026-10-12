"""CLI voice companion — mic → Whisper → Cerebras/Ollama → Fish Audio → speakers.

Usage:
    python src/she.py
    python src/she.py path/to/config/dir

Commands (at the > prompt):
    Enter               record voice
    text + Enter        send text message
    /remember <fact>    save a fact to memory
    /forget             clear all memory facts
    /mute               toggle voice on/off
    /volume <0-1>       set playback volume
    /speed <0.5-2.0>    set TTS speed (1.0 = normal)
    /reload             hot-reload personality.json
    /stats              show token / memory / relationship stats
    /t <seconds>        change max recording duration
    /quit               exit
"""
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# Load .env before any core imports that read API keys at import time 
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()  # fallback: default discovery (CWD, home, etc.


import re
import sys
import time
import random
import logging
import datetime
import subprocess
import webbrowser
import threading
from typing import Dict, List, Optional, Tuple

from faster_whisper import WhisperModel
from fishaudio import FishAudio

from core.audio import Speaker
from core.config import Config
from core.memory import MemoryManager
from core.model import ModelInterface, parse_mood_tag, split_sentences, next_fallback
from core.recording import record_until_silence, WakeWordDetector
logger = logging.getLogger(__name__)


class VoiceCompanion:
    """Main voice companion: ties all core modules together."""

    def __init__(self, config: Config):
        self.config = config
        self.history: List[Dict[str, str]] = []
        self.running = True
        self.max_record_seconds = 20.0

        #  Memory
        self.memory = MemoryManager(
            config_dir=config.config_dir,
            tiers=config.closeness_tiers,
            closeness_enabled=config.closeness_enabled,
        )
        raw_facts = config.load_memory()
        self.memory.load_facts(raw_facts)

        #  Whisper 
        model, device, compute = config.resolve_whisper_settings()
        logger.info(f"Loading Whisper ({model})…")
        self.whisper = WhisperModel(
            model,
            device=device,
            compute_type=compute,
        )
        logger.info(f"Whisper ready: {model} on {device}")

        #  Fish Audio / Speaker 
        logger.info("Initialising Fish Audio client…")
        self.fish_client = FishAudio()
        self.speaker = Speaker(
            client=self.fish_client,
            voice_id=config.preset_voice_id,
            model=config.tts_model,
            emotion_template=config.emotion_template,
            volume=config.volume,
            speed=config.voice_speed,
            cache_enabled=config.tts_cache_enabled,
            cache_max_size=config.tts_cache_max_size,
            ducking_enabled=config.audio_ducking_enabled,
            ducking_volume=config.audio_ducking_volume,
        )

        #  LLM 
        logger.info("Initialising model interface…")
        model_name = config.cerebras_model if config.use_cerebras else config.local_model
        self.model = ModelInterface(
            model_name=model_name,
            temperature=config.persona["temperature"],
            use_cerebras=config.use_cerebras,
            max_tokens=config.persona.get("max_tokens", 300),
            repeat_penalty=config.persona.get("repeat_penalty", 1.1),
            stop=config.persona.get("stop", []),
        )

        #  Wake word 
        self.wake_detector: Optional[WakeWordDetector] = None
        if config.wake_word_enabled:
            self.wake_detector = WakeWordDetector(
                wake_word=config.wake_word,
                sample_rate=config.sample_rate,
                transcribe_fn=self.transcribe,
            )
            logger.info(f"Wake word enabled: '{config.wake_word}'")

        #  Follow-up context + fact extraction 
        # Stores the most recent (user_text, reply_text). Injected into the system
        # prompt below so she can reference continuity without the whole history.
        self.last_exchange: Optional[tuple] = None
        self._turns_since_fact_scan: int = 0
        self._FACT_SCAN_EVERY = 2   # every N turns, to limit API cost on CLI

        logger.info("Initialisation complete")

    # Shutdown 

    def shutdown(self) -> None:
        logger.info("Shutting down…")
        self.speaker.shutdown(timeout=10.0)
        self.config.save_memory(self.memory.get_facts())
        logger.info("Shutdown complete")

    #  Hot-reload config push 

    def apply_config(self) -> None:
        """Push hot-reloaded personality values into the live model and speaker.

        Called after config.reload() so temperature, max_tokens, speed, etc.
        take effect immediately without restarting.
        """
        p = self.config.persona
        # Model
        self.model.temperature    = p.get("temperature", self.model.temperature)
        self.model.max_tokens     = p.get("max_tokens", self.model.max_tokens)
        self.model.repeat_penalty = p.get("repeat_penalty", self.model.repeat_penalty)
        new_stop = p.get("stop", [])
        if self.config.use_cerebras and len(new_stop) > 4:
            new_stop = new_stop[:4]
        self.model.stop = new_stop
        # Speaker
        self.speaker.set_speed(self.config.voice_speed)
        self.speaker.set_emotion_template(self.config.emotion_template)
        self.speaker.set_ducking(
            self.config.audio_ducking_enabled,
            self.config.audio_ducking_volume,
        )
        logger.info("apply_config: model + speaker updated from reloaded personality")

    #  Transcription 

    def transcribe(self, audio) -> str:
        try:
            # Part 3-E: use only the character name as initial_prompt.
            # Including recent assistant replies biases Whisper toward her vocabulary
            # and can cause it to hallucinate her words into the user's utterance.
            initial_prompt = self.config.persona["name"]

            segments, info = self.whisper.transcribe(
                audio,
                language=self.config.whisper_language,
                beam_size=self.config.whisper_beam_size,
                vad_filter=self.config.whisper_vad_filter,
                initial_prompt=initial_prompt,
                condition_on_previous_text=self.config.whisper_condition_on_previous,
                # Part 3-E: pass the default fallback schedule instead of 0.0 alone,
                # so compression_ratio and log_prob thresholds can actually fire.
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                no_speech_threshold=0.6,
            )
            # Part 3-E: segment .text already carries a leading space; join without sep
            text = "".join(s.text for s in segments).strip()
            logger.debug(f"Transcribed: {text[:60]!r}")
            return text
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    #  System message 

    def system_message(self) -> Dict[str, str]:
        raw = self.config.persona["system_prompt"]
        text = "\n\n".join(raw) if isinstance(raw, list) else raw
        text += f" Keep replies under {self.config.persona['max_reply_length']} characters."

        if self.config.persona.get("mood_tags"):
            moods = ", ".join(self.config.persona.get("moods", []))
            text += (
                f" Begin every reply with one mood in square brackets, chosen from: {moods}."
            )

        reminder = self.config.persona.get("reminder", "")
        if reminder:
            text += f"\n\n{reminder}"

        # Follow-up context: remind her of the last thing that was said so she
        # can reference continuity without needing the full rolling history.
        if self.last_exchange:
            user_last, assist_last = self.last_exchange
            text += (
                f"\n\nLast exchange before this turn\n"
                f"User said: {user_last}\nYou said: {assist_last}"
            )

        # Memory context block (facts + closeness hint + episode summaries)
        ctx = self.memory.build_context_block(max_facts=self.config.max_memory_facts)
        if ctx:
            text += f"\n\n{ctx}"

        return {"role": "system", "content": text}

    #  History management with summarization 

    def _trim_history(self) -> None:
        """Keep enough history for both the context window and summarisation.

        Plain `history[-max_history:]` would silently drop messages before they
        are ever summarised when max_history < 2 * summarize_after.
        """
        keep = max(self.config.max_history, self.config.summarize_after * 2)
        self.history = self.history[-keep:]

    def _maybe_summarize(self) -> None:
        """If history exceeds summarize_after, compress the oldest half."""
        limit = self.config.summarize_after
        if len(self.history) < limit * 2:
            return

        to_summarize = self.history[:limit]
        self.history = self.history[limit:]

        logger.info(f"Summarising {len(to_summarize)} old messages…")
        summary = self.model.summarize(to_summarize, self.config.persona["name"])
        if summary:
            today = datetime.date.today().isoformat()
            self.memory.save_summary(today, summary)
            logger.info("Summary saved to episode store")

    # Commands 

    def try_command(self, text: str) -> Optional[str]:
        t = text.lower()

        # 3c: no leading zero on the hour
        if re.search(r"what time is it|what('s| is) the time", t):
            now = datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
            return f"It's {now}."

        m = re.search(r"timer for (\d+)\s*(second|minute|hour)", t)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            secs = n * {"second": 1, "minute": 60, "hour": 3600}[unit]

            def _fire():
                msg = f"Your {n} {unit} timer is up!"
                self.speaker.say(msg)
                logger.info(f"Timer fired: {msg}")

            # 3a: daemon timer so it doesn't keep the process alive after quit
            timer = threading.Timer(secs, _fire)
            timer.daemon = True
            timer.start()
            return f"Okay, timer set for {n} {unit}{'s' if n != 1 else ''}."

        m = re.search(r"\bopen (\w+)", t)
        if m:
            name = m.group(1)
            if name in self.config.apps:
                # 3b: catch OSError so a missing executable doesn't crash the loop
                try:
                    subprocess.Popen(self.config.apps[name])
                except OSError as e:
                    logger.error(f"Could not open app '{name}': {e}")
                    return f"Sorry, I couldn't open {name}."
                return f"Opening {name}."
            if name in self.config.sites:
                webbrowser.open(self.config.sites[name])
                return f"Opening {name}."

        return None

    #  LLM streaming 

    def stream_reply(self, messages: List[Dict[str, str]]) -> Tuple[str, Optional[str]]:
        """Stream model reply, speaking incrementally. Returns (full_text, mood)."""
        full = buf = head = ""
        mood: Optional[str] = None
        mood_done = False

        print(f" {self.config.persona['name']}: ", end="", flush=True)

        try:
            for chunk in self.model.stream_chat(messages):
                piece = chunk["message"]["content"]

                if not mood_done:
                    head += piece
                    parsed_mood, remaining = parse_mood_tag(head)
                    if parsed_mood is not None:
                        mood = parsed_mood
                        piece, mood_done = remaining, True
                    elif len(head.lstrip()) > 30:
                        piece, mood_done = head, True
                    else:
                        continue

                full += piece
                buf += piece
                print(piece, end="", flush=True)

                parts = split_sentences(buf, min_length=20)
                if len(parts) == 2:
                    complete, buf = parts
                    self.speaker.say(complete, mood)

            if not mood_done:
                full = buf = head
            if buf:
                self.speaker.say(buf, mood)

            print()
            return full.strip(), mood

        except Exception as e:
            logger.error(f"Model streaming failed: {e}", exc_info=True)
            fallback = next_fallback()
            print(f"\n  ({fallback})")
            self.speaker.say(fallback)
            if full:
                return full.strip(), mood
            raise

    #  Turn handling 

    def handle_user_input(self, text: str) -> None:
        print(f"👤 You: {text}")

        reply = self.try_command(text)
        if reply:
            print(f" {self.config.persona['name']}: {reply}")
            self.speaker.say(reply)
            self.memory.log_turn("user", text)
            self.memory.log_turn("assistant", reply)
            self.last_exchange = (text, reply)
            return

        self.history.append({"role": "user", "content": text})
        self._maybe_summarize()
        messages = [self.system_message()] + self.history[-self.config.max_history:]

        try:
            reply_text, mood = self.stream_reply(messages)
            self.history.append({"role": "assistant", "content": reply_text})
            self._trim_history()
            self.memory.log_turn("user", text)
            self.memory.log_turn("assistant", reply_text, mood)
            self.memory.after_turn()
            self.last_exchange = (text, reply_text)
            self._extract_facts(text, reply_text)
        except Exception as e:
            logger.error(f"Turn failed: {e}")
            self.history.pop()

    #  Intelligence helpers 

    def _extract_facts(self, user_text: str, reply_text: str) -> None:
        """Ask the model to pull durable facts about the user out of a turn.

        Runs every few turns (not each turn) to limit token/API cost. Facts are
        deduped against existing memory and persisted via config.save_memory.
        """
        if not self.config.auto_extract_facts:
            return
        self._turns_since_fact_scan += 1
        if self._turns_since_fact_scan < self._FACT_SCAN_EVERY:
            return
        self._turns_since_fact_scan = 0

        try:
            prompt = (
                "Extract any durable, factual statements the USER made about "
                "themselves in this exchange (preferences, plans, people, dates, "
                "likes, dislikes, etc.). Each fact on its own line. If there are "
                "none, output nothing. Do not invent or infer — only report what "
                "the user explicitly said. No labels, no numbering.\n\n"
                f"User: {user_text}\nYou: {reply_text}"
            )
            full = ""
            for chunk in self.model.stream_chat(
                [{"role": "user", "content": prompt}], temperature=0.1
            ):
                full += chunk["message"]["content"]
            known = self.memory.get_facts()
            added = 0
            for ln in full.splitlines():
                ln = ln.strip().lstrip("-•")
                if ln and ln not in known:
                    self.memory.add_fact(ln)
                    known.append(ln)
                    added += 1
            if added:
                self.config.save_memory(self.memory.get_facts())
                logger.info(f"Auto-extracted {added} new fact(s)")
        except Exception as e:
            logger.debug(f"Fact extraction skipped: {e}")

    _MISSED_PHRASES = [
        "Sorry, I didn't catch that. Say it again?",
        "Huh? You went quiet on me. Try once more?",
        "I missed that, baby. One more time?",
    ]

    def _missed_ack(self) -> None:
        """Speak a gentle acknowledgment when a record/transcription comes up empty."""
        if not getattr(self.config, "gentle_missed", True):
            return
        if not self.speaker.enabled:
            print("(didn't catch that)")
            return
        self.speaker.say(random.choice(self._MISSED_PHRASES))

    #  CLI commands 

    def cmd_remember(self, fact: str) -> None:
        if not fact:
            print("Usage: /remember <fact>")
            return
        self.memory.add_fact(fact)
        self.config.save_memory(self.memory.get_facts())
        print(f"✓ Remembered: {fact}")

    def cmd_forget(self) -> None:
        self.memory.clear_facts()
        self.config.save_memory([])
        print("✓ Memory facts cleared")

    def cmd_mute(self) -> None:
        self.speaker.enabled = not self.speaker.enabled
        print(f" Voice {'off' if not self.speaker.enabled else 'on'}")

    def cmd_volume(self, arg: str) -> None:
        try:
            self.speaker.set_volume(float(arg))
            print(f" Volume: {self.speaker.volume:.0%}")
        except ValueError:
            print("Usage: /volume <0.0–1.0>")

    def cmd_speed(self, arg: str) -> None:
        try:
            self.speaker.set_speed(float(arg))
            print(f" Speed: {self.speaker.speed:.1f}x")
        except ValueError:
            print("Usage: /speed <0.5–2.0>")

    def cmd_timer(self, arg: str) -> None:
        try:
            self.max_record_seconds = float(arg)
            print(f"⏱  Max recording: {self.max_record_seconds}s")
        except ValueError:
            print("Usage: /t <seconds>")

    def cmd_reload(self) -> None:
        try:
            self.config.reload()
            self.apply_config()
            print(f"✓ Personality reloaded: {self.config.persona['name']}")
        except Exception as e:
            print(f"✗ Reload failed: {e}")

    def cmd_stats(self) -> None:
        tokens = self.model.get_token_stats()
        print(f"\n Statistics:")
        print(f"   Closeness: {self.memory.closeness_score()} pts  ({self.memory.tier_name()})")
        print(f"   Memory facts: {len(self.memory.get_facts())}")
        print(f"   History: {len(self.history)} messages")
        print(f"   Tokens: {tokens['prompt']} prompt  {tokens['completion']} completion")
        backend = "Cerebras" if self.config.use_cerebras else "Ollama"
        print(f"   Backend: {backend} / {self.model.model_name}")
        if self.speaker.cache:
            print(f"   TTS cache: {len(self.speaker.cache.cache)}/{self.speaker.cache.max_size}")

    #  Main loop 

    def run(self) -> None:
        # Greeting — pick randomly from the greetings list each startup
        greeting = self.config.pick_greeting()
        if greeting:
            print(f" {self.config.persona['name']}: {greeting}")
            self.speaker.say(greeting)
            self.memory.log_turn("assistant", greeting)

        print("\nCommands:")
        print("  Enter              record voice")
        print("  text + Enter       text message")
        print("  /remember <fact>   /forget   /mute   /volume <0-1>")
        print("  /speed <0.5-2.0>   /reload   /stats  /t <sec>  /quit\n")

        while self.running:
            try:
                woke = False
                if self.wake_detector:
                    print("💤 Waiting for wake word…")
                    self.wake_detector.wait_for_wake()
                    print("⚡ Wake word detected!")
                    woke = True

                # In wake-word mode text commands are unavailable; cmd is always ""
                # which falls straight through to voice recording below.
                cmd = "" if woke else input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            self.speaker.stop()

            if cmd.startswith("/"):
                name, _, arg = cmd.partition(" ")
                if name == "/quit":
                    break
                elif name == "/mute":
                    self.cmd_mute()
                elif name == "/forget":
                    self.cmd_forget()
                elif name == "/remember":
                    self.cmd_remember(arg)
                elif name in ("/volume", "/v"):
                    self.cmd_volume(arg)
                elif name == "/speed":
                    self.cmd_speed(arg)
                elif name in ("/t", "/timer"):
                    self.cmd_timer(arg)
                elif name == "/reload":
                    self.cmd_reload()
                elif name == "/stats":
                    self.cmd_stats()
                else:
                    print(f" Unknown command: {name}")
                continue

            if cmd:
                self.handle_user_input(cmd)
                continue

            # Voice recording
            print("🎤 Listening…")
            audio = record_until_silence(
                sample_rate=self.config.sample_rate,
                max_seconds=self.max_record_seconds,
                threshold_multiplier=self.config.vad_threshold_multiplier,
            )
            if audio is None or len(audio) / self.config.sample_rate < self.config.min_audio_length:
                self._missed_ack()
                continue

            text = self.transcribe(audio)
            if not text:
                self._missed_ack()
                continue

            self.handle_user_input(text)

        print("\nBye!")


def main() -> None:
    config_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    try:
        config = Config(config_dir=config_dir)
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"[Config error] {e}")
        sys.exit(1)

    companion = VoiceCompanion(config)
    try:
        companion.run()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        companion.shutdown()


if __name__ == "__main__":
    main()
