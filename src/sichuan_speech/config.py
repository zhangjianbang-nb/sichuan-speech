"""Environment-driven configuration.

Every machine-specific path lives here. Nothing else in the package hard-codes
``/home/whu`` or ``~/chuan``. Defaults match the reference training machine
(RTX 4090, GPU 1) so that existing deployments keep working unchanged; override
them with ``SICHUAN_*`` environment variables on any other host.

Example::

    export SICHUAN_GPU=0
    export SICHUAN_SFT_DIR=/models/cosyvoice2-sichuan
    export SICHUAN_DATA_ROOT=/datasets/chuan
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

GPU_RE = re.compile(r"^\d+(,\d+)*$")

_PATH_FIELDS = (
    "cosyvoice_home",
    "base_model_dir",
    "sft_model_dir",
    "asr_base_dir",
    "asr_ckpt",
    "data_root",
    "work_root",
    "eval_root",
    "prompt_wav",
    "official_prompt_wav",
)


def expand_path(value: str | os.PathLike[str]) -> Path:
    """Expand ``~`` and ``$VAR`` and make the path absolute."""
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(str(value)))))


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration."""

    gpu: str = "1"
    host: str = "0.0.0.0"
    port: int = 8100
    log_level: str = "INFO"

    cosyvoice_home: Path = Path("~/CosyVoice")
    base_model_dir: Path = Path("~/models/CosyVoice2-0.5B")
    sft_model_dir: Path = Path("~/chuan/models/cosyvoice2-sichuan2")
    asr_base_dir: Path = Path("~/models/SenseVoiceSmall")
    asr_ckpt: Path = Path("~/chuan/models/sensevoice-sichuan.pt")

    data_root: Path = Path("~/chuan/data")
    work_root: Path = Path("~/chuan")
    eval_root: Path = Path("~/chuan/eval")

    prompt_wav: Path = Path("~/chuan/data/raw/magic_tts/1776909066881_g4plqjyx_32946.wav")
    prompt_text: str = "领导想安排我转岗，我礼貌拒绝，我更想在现在的岗位发展。"
    official_prompt_wav: Path = Path("~/CosyVoice/asset/zero_shot_prompt.wav")
    official_prompt_text: str = "希望你以后能够做的比我还好呦。"

    tts_fp16: bool = True
    asr_device: str = "cpu"

    max_text_chars: int = 500
    max_upload_mb: int = 20
    default_n_samples: int = 5
    max_n_samples: int = 5
    speed_min: float = 0.5
    speed_max: float = 2.0
    request_timeout: float = 180.0
    cors_origins: tuple[str, ...] = ("*",)

    def __post_init__(self) -> None:
        for name in _PATH_FIELDS:
            object.__setattr__(self, name, expand_path(getattr(self, name)))
        if not GPU_RE.match(str(self.gpu)):
            raise ValueError(f"gpu must be like '0' or '0,1', got {self.gpu!r}")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError(f"port out of range: {self.port}")
        if self.max_n_samples < 1:
            raise ValueError("max_n_samples must be >= 1")
        if not 1 <= self.default_n_samples <= self.max_n_samples:
            raise ValueError("default_n_samples must be within [1, max_n_samples]")
        if not 0 < self.speed_min <= self.speed_max:
            raise ValueError("speed_min must be > 0 and <= speed_max")
        if self.max_text_chars < 1:
            raise ValueError("max_text_chars must be >= 1")
        if self.max_upload_mb < 1:
            raise ValueError("max_upload_mb must be >= 1")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``SICHUAN_*`` variables (with legacy aliases)."""
        source: Mapping[str, str] = os.environ if env is None else env

        def pick(*names: str, default: str | None = None) -> str | None:
            for name in names:
                value = source.get(name)
                if value not in (None, ""):
                    return value
            return default

        origins = pick("SICHUAN_CORS_ORIGINS", default="*") or "*"
        return cls(
            gpu=pick("SICHUAN_GPU", "CUDA_VISIBLE_DEVICES", default="1") or "1",
            host=pick("SICHUAN_HOST", "HOST", default="0.0.0.0") or "0.0.0.0",
            port=int(pick("SICHUAN_PORT", "PORT", default="8100") or "8100"),
            log_level=(pick("SICHUAN_LOG_LEVEL", default="INFO") or "INFO").upper(),
            cosyvoice_home=Path(
                pick("SICHUAN_COSYVOICE_HOME", default="~/CosyVoice") or "~/CosyVoice"
            ),
            base_model_dir=Path(
                pick("SICHUAN_BASE_DIR", "BASE_DIR", default="~/models/CosyVoice2-0.5B")
                or "~/models/CosyVoice2-0.5B"
            ),
            sft_model_dir=Path(
                pick("SICHUAN_SFT_DIR", "SFT_DIR", default="~/chuan/models/cosyvoice2-sichuan2")
                or "~/chuan/models/cosyvoice2-sichuan2"
            ),
            asr_base_dir=Path(
                pick("SICHUAN_ASR_BASE_DIR", default="~/models/SenseVoiceSmall")
                or "~/models/SenseVoiceSmall"
            ),
            asr_ckpt=Path(
                pick("SICHUAN_ASR_CKPT", "ASR_CKPT", default="~/chuan/models/sensevoice-sichuan.pt")
                or "~/chuan/models/sensevoice-sichuan.pt"
            ),
            data_root=Path(pick("SICHUAN_DATA_ROOT", default="~/chuan/data") or "~/chuan/data"),
            work_root=Path(pick("SICHUAN_WORK_ROOT", default="~/chuan") or "~/chuan"),
            eval_root=Path(pick("SICHUAN_EVAL_ROOT", default="~/chuan/eval") or "~/chuan/eval"),
            prompt_wav=Path(
                pick(
                    "SICHUAN_PROMPT_WAV",
                    default="~/chuan/data/raw/magic_tts/1776909066881_g4plqjyx_32946.wav",
                )
                or "~/chuan/data/raw/magic_tts/1776909066881_g4plqjyx_32946.wav"
            ),
            prompt_text=pick(
                "SICHUAN_PROMPT_TEXT",
                default="领导想安排我转岗，我礼貌拒绝，我更想在现在的岗位发展。",
            )
            or "领导想安排我转岗，我礼貌拒绝，我更想在现在的岗位发展。",
            official_prompt_wav=Path(
                pick(
                    "SICHUAN_OFFICIAL_PROMPT_WAV", default="~/CosyVoice/asset/zero_shot_prompt.wav"
                )
                or "~/CosyVoice/asset/zero_shot_prompt.wav"
            ),
            official_prompt_text=pick(
                "SICHUAN_OFFICIAL_PROMPT_TEXT", default="希望你以后能够做的比我还好呦。"
            )
            or "希望你以后能够做的比我还好呦。",
            tts_fp16=_as_bool(pick("SICHUAN_TTS_FP16", default="1"), default=True),
            asr_device=pick("SICHUAN_ASR_DEVICE", default="cpu") or "cpu",
            max_text_chars=int(pick("SICHUAN_MAX_TEXT_CHARS", default="500") or "500"),
            max_upload_mb=int(pick("SICHUAN_MAX_UPLOAD_MB", default="20") or "20"),
            default_n_samples=int(pick("SICHUAN_DEFAULT_N_SAMPLES", default="5") or "5"),
            max_n_samples=int(pick("SICHUAN_MAX_N_SAMPLES", default="5") or "5"),
            speed_min=float(pick("SICHUAN_SPEED_MIN", default="0.5") or "0.5"),
            speed_max=float(pick("SICHUAN_SPEED_MAX", default="2.0") or "2.0"),
            request_timeout=float(pick("SICHUAN_REQUEST_TIMEOUT", default="180") or "180"),
            cors_origins=tuple(part.strip() for part in origins.split(",") if part.strip()),
        )

    def with_overrides(self, **kwargs: Any) -> Settings:
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)

    def configure_gpu(self) -> None:
        """Export ``CUDA_VISIBLE_DEVICES``. Call before importing torch."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

    def describe(self) -> dict[str, Any]:
        """Human-readable snapshot used by ``sichuan-speech info`` and ``/api``."""
        return {
            "gpu": self.gpu,
            "host": self.host,
            "port": self.port,
            "cosyvoice_home": str(self.cosyvoice_home),
            "base_model_dir": str(self.base_model_dir),
            "sft_model_dir": str(self.sft_model_dir),
            "asr_base_dir": str(self.asr_base_dir),
            "asr_ckpt": str(self.asr_ckpt),
            "data_root": str(self.data_root),
            "work_root": str(self.work_root),
            "eval_root": str(self.eval_root),
            "prompt_wav": str(self.prompt_wav),
            "official_prompt_wav": str(self.official_prompt_wav),
            "max_text_chars": self.max_text_chars,
            "max_upload_mb": self.max_upload_mb,
            "default_n_samples": self.default_n_samples,
            "max_n_samples": self.max_n_samples,
            "speed_min": self.speed_min,
            "speed_max": self.speed_max,
        }

    def existing(self) -> dict[str, bool]:
        """Which configured paths actually exist on this host."""
        return {name: Path(getattr(self, name)).exists() for name in _PATH_FIELDS}


def _as_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
