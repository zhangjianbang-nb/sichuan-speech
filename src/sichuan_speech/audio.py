"""Audio decoding, resampling and WAV encoding.

``soundfile`` handles WAV/FLAC/OGG natively. Anything else (mp3, m4a, webm/opus
recorded by browsers) is handed to ``ffmpeg`` so that the ASR endpoint accepts
whatever a user drags into the UI. Resampling prefers ``scipy.signal.resample_poly``
and falls back to a windowed-sinc implementation so the package has no hard
scipy dependency.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from math import gcd
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

TARGET_SR = 16000
DEFAULT_SUBTYPE = "PCM_16"


class AudioDecodeError(RuntimeError):
    """Raised when neither libsndfile nor ffmpeg can decode the input."""


def _soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise AudioDecodeError(
            "soundfile is required for audio I/O; install it with `pip install soundfile`"
        ) from exc
    return sf


def to_mono(wav: np.ndarray) -> np.ndarray:
    """Downmix to mono float32."""
    array = np.asarray(wav, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return array


def load_audio(
    source: str | Path | bytes,
    *,
    target_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """Decode ``source`` (path or raw bytes) to mono float32.

    Returns ``(samples, sample_rate)``. When ``target_sr`` is given the audio is
    resampled to it.
    """
    if isinstance(source, (bytes, bytearray)):
        wav, sr = _decode_bytes(bytes(source))
    else:
        path = Path(source)
        if not path.exists():
            raise AudioDecodeError(f"audio file not found: {path}")
        try:
            sf = _soundfile()
            wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception as exc:
            logger.debug("soundfile failed on %s (%s), trying ffmpeg", path, exc)
            wav, sr = _decode_with_ffmpeg(bytes(path.read_bytes()))
    wav = to_mono(wav)
    if target_sr is not None and sr != target_sr:
        wav = resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def _decode_bytes(data: bytes) -> tuple[np.ndarray, int]:
    try:
        sf = _soundfile()
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return wav, sr
    except Exception as exc:
        logger.debug("soundfile failed on %d bytes (%s), trying ffmpeg", len(data), exc)
        return _decode_with_ffmpeg(data)


def _decode_with_ffmpeg(data: bytes) -> tuple[np.ndarray, int]:
    if shutil.which("ffmpeg") is None:
        raise AudioDecodeError(
            "cannot decode this audio format: install ffmpeg or upload a WAV/FLAC file"
        )
    with tempfile.TemporaryDirectory(prefix="sichuan-audio-") as tmp:
        src = os.path.join(tmp, "input.bin")
        dst = os.path.join(tmp, "output.wav")
        Path(src).write_bytes(data)
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                src,
                "-ac",
                "1",
                "-ar",
                str(TARGET_SR),
                "-f",
                "wav",
                dst,
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0 or not os.path.exists(dst):
            message = proc.stderr.decode("utf-8", "replace").strip() or "unknown ffmpeg error"
            raise AudioDecodeError(f"ffmpeg could not decode the audio: {message}")
        sf = _soundfile()
        wav, sr = sf.read(dst, dtype="float32", always_2d=False)
    return wav, sr


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample mono float32 audio."""
    array = to_mono(wav)
    if sr_in == sr_out or array.size == 0:
        return array
    try:
        from scipy.signal import resample_poly

        divisor = gcd(int(sr_in), int(sr_out))
        out = resample_poly(array, int(sr_out) // divisor, int(sr_in) // divisor)
        return np.asarray(out, dtype=np.float32)
    except Exception:
        logger.debug("scipy unavailable, using windowed-sinc resampler")
        return _sinc_resample(array, sr_in, sr_out)


def _sinc_resample(wav: np.ndarray, sr_in: int, sr_out: int, half_taps: int = 16) -> np.ndarray:
    """Windowed-sinc resampler (Blackman-windowed, 32 taps).

    Good enough for 16 kHz ASR input; not intended for music production.
    """
    ratio = float(sr_out) / float(sr_in)
    n_out = round(len(wav) * ratio)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    cutoff = min(1.0, ratio)
    offsets = np.arange(-half_taps + 1, half_taps + 1, dtype=np.float64)
    positions = np.arange(n_out, dtype=np.float64) / ratio
    base = np.floor(positions).astype(np.int64)
    frac = positions - base

    out = np.empty(n_out, dtype=np.float32)
    block = 16384
    window = np.blackman(2 * half_taps)
    for start in range(0, n_out, block):
        stop = min(start + block, n_out)
        f = frac[start:stop, None]
        kernel = cutoff * np.sinc(cutoff * (offsets[None, :] - f)) * window[None, :]
        kernel /= kernel.sum(axis=1, keepdims=True)
        indices = base[start:stop, None] + offsets[None, :].astype(np.int64)
        valid = (indices >= 0) & (indices < len(wav))
        clipped = np.clip(indices, 0, len(wav) - 1)
        values = wav[clipped] * kernel * valid
        out[start:stop] = values.sum(axis=1)
    return out


def duration(wav: np.ndarray, sr: int) -> float:
    """Duration in seconds."""
    return float(len(wav)) / float(sr) if sr else 0.0


def wav_bytes(wav: np.ndarray, sr: int, *, subtype: str = DEFAULT_SUBTYPE) -> bytes:
    """Encode mono float32 audio as a WAV byte string."""
    sf = _soundfile()
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(wav, dtype=np.float32), int(sr), format="WAV", subtype=subtype)
    return buffer.getvalue()


def write_wav(
    path: str | Path, wav: np.ndarray, sr: int, *, subtype: str = DEFAULT_SUBTYPE
) -> Path:
    """Write a WAV file, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf = _soundfile()
    sf.write(str(target), np.asarray(wav, dtype=np.float32), int(sr), subtype=subtype)
    return target


def concat_wavs(
    paths: Sequence[str | Path],
    *,
    gap_s: float = 0.35,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    """Concatenate clips with silence between them, resampling to a common rate."""
    if not paths:
        raise ValueError("concat_wavs needs at least one path")
    rate = sample_rate
    pieces: list[np.ndarray] = []
    for path in paths:
        wav, sr = load_audio(path)
        if rate is None:
            rate = sr
        elif sr != rate:
            wav = resample(wav, sr, rate)
        pieces.append(wav)
        if gap_s > 0:
            pieces.append(np.zeros(int(rate * gap_s), dtype=np.float32))
    if gap_s > 0:
        pieces.pop()
    assert rate is not None
    return np.concatenate(pieces), int(rate)


def peak_normalize(wav: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Scale so the loudest sample is ``peak``; no-op for silence."""
    array = np.asarray(wav, dtype=np.float32)
    maximum = float(np.max(np.abs(array))) if array.size else 0.0
    if maximum <= 1e-9:
        return array
    return (array * (peak / maximum)).astype(np.float32)
