"""Audio I/O: decode from path and bytes, resample, write, concatenate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sichuan_speech.audio import (
    TARGET_SR,
    AudioDecodeError,
    concat_wavs,
    duration,
    load_audio,
    peak_normalize,
    resample,
    to_mono,
    wav_bytes,
    write_wav,
)


def test_write_then_load_roundtrip(tmp_path: Path, sine_wav):
    wav, sr = sine_wav
    path = write_wav(tmp_path / "a.wav", wav, sr)
    loaded, loaded_sr = load_audio(path, target_sr=TARGET_SR)
    assert loaded_sr == TARGET_SR
    assert loaded.shape == wav.shape
    assert np.max(np.abs(loaded - wav)) < 1e-3


def test_load_audio_accepts_raw_bytes(tmp_path: Path, sine_wav):
    wav, sr = sine_wav
    payload = wav_bytes(wav, sr)
    loaded, loaded_sr = load_audio(payload, target_sr=TARGET_SR)
    assert loaded_sr == TARGET_SR
    assert loaded.shape == wav.shape


def test_wav_bytes_starts_with_riff(sine_wav):
    wav, sr = sine_wav
    payload = wav_bytes(wav, sr)
    assert payload[:4] == b"RIFF"
    assert payload[8:12] == b"WAVE"


def test_load_audio_resamples_to_target(tmp_path: Path):
    sample_rate = 8000
    wav = (0.1 * np.sin(np.arange(sample_rate) * 0.05)).astype(np.float32)
    path = write_wav(tmp_path / "8k.wav", wav, sample_rate)
    loaded, loaded_sr = load_audio(path, target_sr=TARGET_SR)
    assert loaded_sr == TARGET_SR
    assert abs(len(loaded) / TARGET_SR - 1.0) < 0.02


def test_load_audio_rejects_garbage():
    with pytest.raises(AudioDecodeError):
        load_audio(b"not audio at all", target_sr=TARGET_SR)


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(10, dtype=np.float32), np.zeros(10, dtype=np.float32)], axis=1)
    mono = to_mono(stereo)
    assert mono.ndim == 1
    assert np.allclose(mono, 0.5)


def test_resample_identity_when_rates_match(sine_wav):
    wav, sr = sine_wav
    assert resample(wav, sr, sr) is wav


def test_resample_changes_length_proportionally(sine_wav):
    wav, sr = sine_wav
    doubled = resample(wav, sr, sr * 2)
    assert abs(len(doubled) - len(wav) * 2) <= 2


def test_duration(sine_wav):
    wav, sr = sine_wav
    assert duration(wav, sr) == pytest.approx(0.5, abs=1e-3)


def test_peak_normalize_scales_loudest_sample(sine_wav):
    wav, _ = sine_wav
    quiet = wav * 0.1
    normalized = peak_normalize(quiet, peak=0.9)
    assert np.max(np.abs(normalized)) == pytest.approx(0.9, abs=1e-4)


def test_concat_wavs_inserts_gap(tmp_path: Path, sine_wav):
    wav, sr = sine_wav
    first = write_wav(tmp_path / "1.wav", wav, sr)
    second = write_wav(tmp_path / "2.wav", wav, sr)
    merged, merged_sr = concat_wavs([first, second], gap_s=0.1)
    assert merged_sr == TARGET_SR
    assert duration(merged, merged_sr) == pytest.approx(1.1, abs=0.02)


def test_concat_wavs_rejects_empty_input():
    with pytest.raises(ValueError):
        concat_wavs([])
