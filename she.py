"""Voice companion: mic -> Whisper (CPU) -> Ollama (GPU) -> Fish Audio -> speakers.

Refactored with proper architecture, logging, error recovery, and graceful shutdown.

pip install numpy sounddevice ollama faster-whisper python-dotenv fishaudio pygame
"""
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import re
import sys
import time
import logging
import datetime
import subprocess
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from fishaudio import FishAudio

from audio import Speaker
from model import ModelInterface, parse_mood_tag, split_sentences
from recording import record_until_silence
from config import Config

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class VoiceCompanion:
    """Main voice companion application."""
    
    def __init__(self, config: Config):
        self.config = config
        self.facts = config.load_memory()
        self.history: List[Dict[str, str]] = []
        
        # Initialize components
        logger.info("Initializing Whisper...")
        self.whisper = WhisperModel("base", device="cpu", compute_type="int8")
        
        logger.info("Initializing Fish Audio client...")
        self.fish_client = FishAudio()
        
        logger.info("Initializing speaker...")
        self.speaker = Speaker(
            client=self.fish_client,
            voice_id=config.preset_voice_id,
            model=config.tts_model,
            emotion_template=config.emotion_template,
            volume=config.volume,
            cache_enabled=config.tts_cache_enabled,
            cache_max_size=config.tts_cache_max_size
        )
        
        logger.info("Initializing model interface...")
        self.model = ModelInterface(
            model_name=config.local_model,
            temperature=config.persona["temperature"]
        )
        
        self.max_record_seconds = 20.0
        self.running = True
        
        logger.info("Initialization complete")
    
    def shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("Shutting down...")
        self.speaker.shutdown(timeout=10.0)
        self.config.save_memory(self.facts)
        logger.info("Shutdown complete")
    
    def transcribe(self, audio) -> str:
        """Transcribe audio using Whisper."""
        try:
            segments, _ = self.whisper.transcribe(
                audio,
                language="en",
                beam_size=1,
                vad_filter=True,
                initial_prompt=self.config.persona["name"],
            )
            text = " ".join(s.text for s in segments).strip()
            logger.debug(f"Transcribed: {text}")
            return text
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
    
    def system_message(self) -> Dict[str, str]:
        """Build system message with memory."""
        raw = self.config.persona["system_prompt"]
        text = ("\n\n".join(raw) if isinstance(raw, list) else raw)
        text += f" Keep replies under {self.config.persona['max_reply_length']} characters."
        
        if self.config.persona.get("mood_tags"):
            text += (" Begin every reply with one mood in square brackets, chosen from: "
                     "joyful, excited, embarrassed, sad, angry, surprised, curious, relaxed.")
        
        if self.facts:
            text += " Things you know about the user: " + "; ".join(self.facts) + "."
        
        return {"role": "system", "content": text}
    
    def try_command(self, text: str) -> Optional[str]:
        """Handle instant commands. Returns response or None."""
        t = text.lower()
        
        # Time query
        if re.search(r"what time is it|what('s| is) the time", t):
            return datetime.datetime.now().strftime("It's %I:%M %p.")
        
        # Timer
        m = re.search(r"timer for (\d+)\s*(second|minute|hour)", t)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            seconds = n * {"second": 1, "minute": 60, "hour": 3600}[unit]
            
            def timer_callback():
                msg = f"Your {n} {unit} timer is up!"
                self.speaker.say(msg)
                logger.info(f"Timer fired: {msg}")
            
            import threading
            threading.Timer(seconds, timer_callback).start()
            return f"Okay, timer set for {n} {unit}{'s' if n != 1 else ''}."
        
        # Open app/site
        m = re.search(r"\bopen (\w+)", t)
        if m:
            name = m.group(1)
            if name in self.config.apps:
                subprocess.Popen(self.config.apps[name])
                return f"Opening {name}."
            if name in self.config.sites:
                webbrowser.open(self.config.sites[name])
                return f"Opening {name}."
        
        return None
    
    def stream_reply(self, messages: List[Dict[str, str]]) -> str:
        """Stream model reply, speaking incrementally. Returns full text."""
        full = buf = head = ""
        mood, mood_done = None, False
        
        print(f" {self.config.persona['name']}: ", end="", flush=True)
        
        try:
            for chunk in self.model.stream_chat(messages):
                piece = chunk["message"]["content"]
                
                # Extract mood tag if present
                if not mood_done:
                    head += piece
                    parsed_mood, remaining = parse_mood_tag(head)
                    if parsed_mood is not None:
                        mood = parsed_mood
                        piece, mood_done = remaining, True
                    elif len(head.lstrip()) > 30:
                        # No mood tag found, proceed
                        piece, mood_done = head, True
                    else:
                        continue
                
                full += piece
                buf += piece
                print(piece, end="", flush=True)
                
                # Split and speak complete sentences
                parts = split_sentences(buf, min_length=20)
                if len(parts) == 2:
                    complete, buf = parts
                    self.speaker.say(complete, mood)
            
            # Handle very short replies or trailing buffer
            if not mood_done:
                full = buf = head
            if buf:
                self.speaker.say(buf, mood)
            
            print()
            return full.strip()
        
        except Exception as e:
            logger.error(f"Model streaming failed: {e}", exc_info=True)
            # Recover partially streamed content
            if full:
                logger.info(f"Recovered partial reply ({len(full)} chars)")
                print(f"\n  (partial reply recovered)")
                return full.strip()
            raise
    
    def handle_user_input(self, text: str) -> None:
        """Process user input and generate reply."""
        print(f"👤 You: {text}")
        
        # Try command first
        reply = self.try_command(text)
        if reply:
            print(f" {self.config.persona['name']}: {reply}")
            self.speaker.say(reply)
            return
        
        # Send to model
        self.history.append({"role": "user", "content": text})
        messages = [self.system_message()] + self.history[-self.config.max_history:]
        
        try:
            reply_text = self.stream_reply(messages)
            self.history.append({"role": "assistant", "content": reply_text})
            self.history = self.history[-self.config.max_history:]
        except Exception as e:
            logger.error(f"Failed to get model reply: {e}")
            print(f"\n  Brain error (is Ollama running?): {e}")
            # Remove user message since we couldn't reply
            self.history.pop()
    
    def cmd_remember(self, fact: str) -> None:
        """Add a fact to memory."""
        if not fact:
            print("Usage: /remember <fact>")
            return
        self.facts.append(fact)
        self.config.save_memory(self.facts)
        print(f"✓ Remembered: {fact}")
        logger.info(f"Added memory fact: {fact}")
    
    def cmd_forget(self) -> None:
        """Clear all memory."""
        self.facts.clear()
        self.config.save_memory(self.facts)
        print("✓ Memory cleared")
        logger.info("Memory cleared")
    
    def cmd_mute(self) -> None:
        """Toggle voice on/off."""
        self.speaker.enabled = not self.speaker.enabled
        status = "off" if not self.speaker.enabled else "on"
        print(f" Voice {status}")
        logger.info(f"Voice {status}")
    
    def cmd_volume(self, arg: str) -> None:
        """Set volume."""
        try:
            vol = float(arg)
            self.speaker.set_volume(vol)
            print(f" Volume: {self.speaker.volume:.0%}")
        except ValueError:
            print("Usage: /volume <0.0-1.0>")
    
    def cmd_timer(self, arg: str) -> None:
        """Set max recording duration."""
        try:
            self.max_record_seconds = float(arg)
            print(f"⏱  Max recording: {self.max_record_seconds}s")
        except ValueError:
            print("Usage: /t <seconds>")
    
    def cmd_stats(self) -> None:
        """Show statistics."""
        tokens = self.model.get_token_stats()
        print(f"\n Statistics:")
        print(f"   Memory facts: {len(self.facts)}")
        print(f"   Conversation history: {len(self.history)} messages")
        print(f"   Total tokens: {tokens['prompt']} prompt + {tokens['completion']} completion")
        if self.speaker.cache:
            print(f"   TTS cache: {len(self.speaker.cache.cache)}/{self.speaker.cache.max_size}")
    
    def run(self) -> None:
        """Main loop."""
        # Greeting
        if self.config.persona.get("greeting"):
            greeting = self.config.persona["greeting"]
            print(f" {self.config.persona['name']}: {greeting}")
            self.speaker.say(greeting)
        
        # Help text
        print("\nCommands:")
        print("  Enter           = record voice")
        print("  type + Enter    = text message")
        print("  /remember <fact>  /forget  /mute  /volume <0-1>  /t <sec>  /stats  /quit")
        
        while self.running:
            try:
                cmd = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            
            # Interrupt any ongoing speech
            self.speaker.stop()
            
            # Handle commands
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
                elif name == "/volume" or name == "/v":
                    self.cmd_volume(arg)
                elif name == "/t":
                    self.cmd_timer(arg)
                elif name == "/stats":
                    self.cmd_stats()
                else:
                    print(f" Unknown command: {name}")
                continue
            
            # Text message
            if cmd:
                self.handle_user_input(cmd)
                continue
            
            # Voice recording
            print("🎤 Listening...")
            audio = record_until_silence(
                sample_rate=self.config.sample_rate,
                max_seconds=self.max_record_seconds,
                threshold_multiplier=self.config.vad_threshold_multiplier
            )
            
            if audio is None or len(audio) / self.config.sample_rate < self.config.min_audio_length:
                print("(didn't catch that)")
                continue
            
            text = self.transcribe(audio)
            if not text:
                print("(didn't catch that)")
                continue
            
            self.handle_user_input(text)
        
        print("\nBye!")


def main():
    """Entry point."""
    
    config_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    
    try:
        config = Config(config_dir=config_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    companion = VoiceCompanion(config)
    
    try:
        companion.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        companion.shutdown()


if __name__ == "__main__":
    main()
