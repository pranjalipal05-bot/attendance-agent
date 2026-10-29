"""
Module 2a - Audio signal processing (offline).

Extracts the audio track with the ffmpeg binary bundled in `imageio-ffmpeg`,
then measures loudness (RMS) once per second. Seconds that are clearly above
the background-noise floor count as "audio activity" (someone is speaking).
No speech is recorded or transcribed - only loudness numbers are kept.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def audio_activity(video_path):
    """Return a boolean array, one entry per second (True = audio activity),
    or None if the video has no usable audio track."""
    exe = _ffmpeg_exe()
    if not exe:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        cmd = [exe, "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", tmp.name]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        rate, data = wavfile.read(tmp.name)
    except Exception:
        return None
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    if data.size == 0:
        return None
    data = data.astype(np.float32)
    peak = np.max(np.abs(data))
    if peak == 0:
        return None
    data /= peak

    n = len(data) // rate
    if n == 0:
        return None
    rms = np.sqrt(np.mean(data[: n * rate].reshape(n, rate) ** 2, axis=1))
    noise_floor = np.percentile(rms, 20)
    threshold = max(noise_floor * 2.0, 0.02)
    return rms > threshold
