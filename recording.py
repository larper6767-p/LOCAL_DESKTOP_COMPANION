"""Audio recording with voice activity detection."""
import time
import logging
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


def rms(block: np.ndarray) -> float:
    """Calculate RMS (root mean square) of audio block."""
    return float(np.sqrt(np.mean(np.square(block))))


def record_until_silence(
    sample_rate: int = 16000,
    max_seconds: float = 20,
    silence_seconds: float = 1.0,
    wait_seconds: float = 8,
    threshold_multiplier: float = 3.0
) -> Optional[np.ndarray]:
    """Record when voice detected, stop after silence. Returns audio or None.
    
    Args:
        sample_rate: Audio sample rate in Hz
        max_seconds: Maximum recording duration
        silence_seconds: Duration of silence before stopping
        wait_seconds: Max time to wait for voice activity to start
        threshold_multiplier: Multiplier for ambient noise to set threshold
    
    Returns:
        Audio array or None if no speech detected
    """
    block = int(sample_rate * 0.03)  # 30ms blocks
    needed_silence = int(silence_seconds / 0.03)
    preroll = deque(maxlen=8)  # keeps the first syllable
    chunks, started, silent = [], False, 0
    
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block
        ) as stream:
            # Calibrate noise floor
            noise_samples = [rms(stream.read(block)[0]) for _ in range(10)]
            noise = np.mean(noise_samples)
            threshold = max(noise * threshold_multiplier, 0.01)
            logger.debug(f"VAD threshold: {threshold:.4f} (noise floor: {noise:.4f})")
            
            t0 = time.time()
            while True:
                data, _ = stream.read(block)
                data = data.copy()
                level = rms(data)
                
                if level > threshold:
                    if not started:
                        started = True
                        chunks.extend(preroll)
                        logger.debug("Voice activity detected")
                    silent = 0
                elif started:
                    silent += 1
                
                if started:
                    chunks.append(data)
                    if silent >= needed_silence:
                        logger.debug(f"Silence detected, stopping (duration: {time.time() - t0:.1f}s)")
                        break
                else:
                    preroll.append(data)
                    if time.time() - t0 > wait_seconds:
                        logger.debug("No voice activity detected within timeout")
                        return None
                
                if time.time() - t0 > max_seconds:
                    logger.debug(f"Max duration reached ({max_seconds}s)")
                    break
        
        if not chunks:
            return None
        
        audio = np.concatenate(chunks).flatten()
        duration = len(audio) / sample_rate
        logger.info(f"Recorded {duration:.1f}s of audio")
        return audio
    
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        return None
