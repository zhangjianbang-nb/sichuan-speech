"""Shared fixtures: put ``src`` on the path and build temporary settings/audio."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sichuan_speech.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing at empty temporary paths."""
    return Settings(
        cosyvoice_home=tmp_path / "CosyVoice",
        base_model_dir=tmp_path / "base",
        sft_model_dir=tmp_path / "sft",
        asr_base_dir=tmp_path / "asr",
        asr_ckpt=tmp_path / "asr.pt",
        data_root=tmp_path / "data",
        work_root=tmp_path / "work",
        eval_root=tmp_path / "eval",
        prompt_wav=tmp_path / "prompt.wav",
        official_prompt_wav=tmp_path / "official.wav",
    )


@pytest.fixture
def ready_settings(settings: Settings) -> Settings:
    """Settings whose model directories and prompt files exist (voices available)."""
    for directory in (settings.sft_model_dir, settings.base_model_dir, settings.asr_base_dir):
        directory.mkdir(parents=True, exist_ok=True)
    settings.prompt_wav.write_bytes(b"RIFF")
    settings.official_prompt_wav.write_bytes(b"RIFF")
    return settings


@pytest.fixture
def sine_wav() -> tuple[np.ndarray, int]:
    """0.5 s of 220 Hz mono float32 audio at 16 kHz."""
    sample_rate = 16000
    time = np.arange(int(0.5 * sample_rate)) / sample_rate
    return (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32), sample_rate
