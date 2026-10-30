"""Settings: defaults, environment parsing, validation and path expansion."""

from __future__ import annotations

from pathlib import Path

import pytest

from sichuan_speech.config import Settings, expand_path


def test_defaults_are_valid():
    settings = Settings()
    assert settings.gpu == "1"
    assert settings.port == 8100
    assert settings.max_n_samples == 5
    assert settings.speed_min <= settings.speed_max


def test_paths_are_expanded_to_absolute():
    settings = Settings()
    for name in ("cosyvoice_home", "base_model_dir", "data_root"):
        value = getattr(settings, name)
        assert isinstance(value, Path)
        assert value.is_absolute()


def test_from_env_reads_primary_names():
    env = {
        "SICHUAN_GPU": "0",
        "SICHUAN_PORT": "9000",
        "SICHUAN_HOST": "127.0.0.1",
        "SICHUAN_SFT_DIR": "/tmp/sft",
        "SICHUAN_ASR_CKPT": "/tmp/asr.pt",
        "SICHUAN_MAX_TEXT_CHARS": "120",
        "SICHUAN_TTS_FP16": "0",
        "SICHUAN_CORS_ORIGINS": "http://localhost:3000",
    }
    settings = Settings.from_env(env)
    assert settings.gpu == "0"
    assert settings.port == 9000
    assert settings.host == "127.0.0.1"
    assert settings.sft_model_dir == Path("/tmp/sft")
    assert settings.asr_ckpt == Path("/tmp/asr.pt")
    assert settings.max_text_chars == 120
    assert settings.tts_fp16 is False
    assert settings.cors_origins == ("http://localhost:3000",)


def test_from_env_falls_back_to_legacy_aliases():
    settings = Settings.from_env({"CUDA_VISIBLE_DEVICES": "0,1", "BASE_DIR": "/tmp/base", "ASR_CKPT": "/tmp/a.pt"})
    assert settings.gpu == "0,1"
    assert settings.base_model_dir == Path("/tmp/base")
    assert settings.asr_ckpt == Path("/tmp/a.pt")


def test_from_env_ignores_empty_values():
    settings = Settings.from_env({"SICHUAN_PORT": "", "SICHUAN_GPU": ""})
    assert settings.port == 8100
    assert settings.gpu == "1"


def test_from_env_rejects_bad_gpu():
    with pytest.raises(ValueError, match="gpu must be"):
        Settings.from_env({"SICHUAN_GPU": "GPU-abc"})


def test_port_out_of_range_rejected():
    with pytest.raises(ValueError, match="port out of range"):
        Settings(port=70000)


def test_default_samples_must_fit_max():
    with pytest.raises(ValueError, match="default_n_samples"):
        Settings(default_n_samples=9, max_n_samples=5)


def test_speed_bounds_validated():
    with pytest.raises(ValueError, match="speed_min"):
        Settings(speed_min=0.0)
    with pytest.raises(ValueError, match="speed_min"):
        Settings(speed_min=2.0, speed_max=1.0)


def test_with_overrides_returns_a_copy():
    settings = Settings()
    changed = settings.with_overrides(port=1234)
    assert changed.port == 1234
    assert settings.port == 8100
    assert changed is not settings


def test_describe_and_existing(tmp_path: Path):
    settings = Settings(data_root=tmp_path / "data", work_root=tmp_path / "work")
    described = settings.describe()
    assert described["port"] == 8100
    assert described["data_root"] == str(tmp_path / "data")
    existing = settings.existing()
    assert existing["data_root"] is False
    assert existing["work_root"] is False


def test_expand_path_handles_vars_and_tilde(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SICHUAN_TEST_DIR", "/tmp/xyz")
    assert expand_path("$SICHUAN_TEST_DIR/sub") == Path("/tmp/xyz/sub")
    assert expand_path("~/sub").is_absolute()
