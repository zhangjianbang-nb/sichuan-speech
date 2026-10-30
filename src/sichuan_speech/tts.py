"""CosyVoice2 engines and the voice registry.

Four voices are exposed by the service:

===============  ================  ==============================================
id               engine            meaning
===============  ================  ==============================================
sichuan          fine-tuned LLM    Chengdu female speaker, no prompt audio
sichuan_avg      fine-tuned LLM    mean speaker embedding of 2233 Sichuan voices
sichuan_zeroshot stock model       Sichuan prompt audio, zero-shot clone (unstable)
mandarin         stock model       Mandarin prompt audio, original capability
===============  ================  ==============================================

Important: CosyVoice2's LLM does **not** consume the speaker embedding (its
``lm_input`` is ``[sos, text_emb, task_id, prompt_speech_token]``). Accent is
decided entirely by the LLM's text-to-speech-token mapping; the embedding only
changes timbre in the flow stage. That is why ``sichuan`` and ``sichuan_avg``
transcribe identically but sound different.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)


class VoiceUnavailable(RuntimeError):
    """Raised when a voice's model directory or prompt audio is missing."""


@dataclass(frozen=True)
class Voice:
    id: str
    label: str
    description: str
    engine: str  # "sft" or "zero_shot"
    speaker: str | None = None
    prompt_wav: Path | None = None
    prompt_text: str | None = None


def build_voices(settings: Settings) -> dict[str, Voice]:
    """The voice registry, resolved against this host's configuration."""
    return {
        "sichuan": Voice(
            id="sichuan",
            label="川话 · 成都女声",
            description="微调模型默认音色，无需提示音频；音色相似度 0.678。",
            engine="sft",
            speaker="sichuan_female",
        ),
        "sichuan_avg": Voice(
            id="sichuan_avg",
            label="川话 · 平均音色",
            description="2233 条川话说话人向量的均值，口音相同、音色更中性。",
            engine="sft",
            speaker="sichuan_avg",
        ),
        "sichuan_zeroshot": Voice(
            id="sichuan_zeroshot",
            label="川话 · 零样本克隆",
            description="原版模型 + 川话提示音，对照用；稳定性差，回读 CER 约 0.25。",
            engine="zero_shot",
            prompt_wav=settings.prompt_wav,
            prompt_text=settings.prompt_text,
        ),
        "mandarin": Voice(
            id="mandarin",
            label="普通话 · 原版女声",
            description="原版模型普通话女声，结构上保证普通话能力不退化。",
            engine="zero_shot",
            prompt_wav=settings.official_prompt_wav,
            prompt_text=settings.official_prompt_text,
        ),
    }


def _ensure_cosyvoice_importable(settings: Settings) -> None:
    """CosyVoice is usually a git checkout, not a pip package."""
    try:
        import cosyvoice  # noqa: F401
    except ImportError:
        for candidate in (
            settings.cosyvoice_home,
            settings.cosyvoice_home / "third_party" / "Matcha-TTS",
        ):
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))
        try:
            import cosyvoice  # noqa: F401
        except ImportError as exc:
            raise VoiceUnavailable(
                "CosyVoice is not importable. Point SICHUAN_COSYVOICE_HOME at the checkout "
                f"(currently {settings.cosyvoice_home}) and make sure its dependencies are installed."
            ) from exc


def check_transformers_compat(cosyvoice_home: Path) -> bool:
    """Warn when transformers>=5 is used with an unpatched CosyVoice checkout.

    ``forward_one_step`` passes a 1-token attention mask during cached decoding;
    transformers 5 expands it literally and masks out every cached position, so
    the model can only see the SOS token and emits gibberish. Fix:
    ``bash scripts/patch_transformers5.sh <cosyvoice_home>``.
    """
    try:
        import transformers
    except ImportError:  # pragma: no cover - transformers is part of the CosyVoice env
        return True
    try:
        major = int(str(transformers.__version__).split(".")[0])
    except ValueError:  # pragma: no cover
        return True
    if major < 5:
        return True
    llm_py = Path(cosyvoice_home) / "cosyvoice" / "llm" / "llm.py"
    if llm_py.exists() and "Rebuild a full-length mask" in llm_py.read_text(
        encoding="utf-8", errors="replace"
    ):
        return True
    logger.warning(
        "transformers %s detected but %s is not patched; TTS output will be gibberish. "
        "Run: bash scripts/patch_transformers5.sh %s",
        transformers.__version__,
        llm_py,
        cosyvoice_home,
    )
    return False


class CosyVoiceEngine:
    """One loaded CosyVoice2 model directory (stock or fine-tuned)."""

    def __init__(self, model_dir: str | Path, *, fp16: bool = True, settings: Settings) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise VoiceUnavailable(f"model directory not found: {self.model_dir}")
        _ensure_cosyvoice_importable(settings)
        check_transformers_compat(settings.cosyvoice_home)
        from cosyvoice.cli.cosyvoice import AutoModel

        logger.info("loading CosyVoice2 from %s (fp16=%s)", self.model_dir, fp16)
        self.model = AutoModel(model_dir=str(self.model_dir), fp16=fp16)
        self.sample_rate = int(self.model.sample_rate)

    def list_speakers(self) -> list[str]:
        return list(self.model.list_available_spks())

    def synthesize(self, text: str, voice: Voice, *, speed: float = 1.0) -> tuple[np.ndarray, int]:
        """Render ``text`` with ``voice``; returns ``(mono float32, sample_rate)``."""
        if voice.engine == "sft":
            speakers = self.list_speakers()
            if voice.speaker not in speakers:
                raise VoiceUnavailable(
                    f"speaker {voice.speaker!r} is not in {self.model_dir.name} "
                    f"(available: {', '.join(speakers) or 'none'})"
                )
            iterator = self.model.inference_sft(
                text, voice.speaker, stream=False, speed=speed
            )
        elif voice.engine == "zero_shot":
            if voice.prompt_wav is None or not Path(voice.prompt_wav).exists():
                raise VoiceUnavailable(f"prompt audio missing: {voice.prompt_wav}")
            iterator = self.model.inference_zero_shot(
                text,
                voice.prompt_text or "",
                str(voice.prompt_wav),
                stream=False,
                speed=speed,
            )
        else:  # pragma: no cover - registry is closed
            raise VoiceUnavailable(f"unknown engine {voice.engine!r}")

        chunks = [out["tts_speech"].squeeze().cpu().numpy() for out in iterator]
        if not chunks:
            raise VoiceUnavailable("the model produced no audio")
        wav = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        return np.asarray(wav, dtype=np.float32), self.sample_rate
