"""SenseVoice recognisers: the stock Mandarin model and the Sichuan fine-tune.

The fine-tuned checkpoint was trained with the label prefix
``[<|zh|>, <|Speech|>, <|NEUTRAL|>, <|woitn|>]`` and therefore only supports
``use_itn=False``. The stock model is evaluated through the same code path (CTC
greedy decode over the rich-transcription head) so both sides of every comparison
share one decoding implementation — the mistake that produced a fake improvement
in round 1 was letting the two paths differ.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .audio import TARGET_SR, load_audio
from .config import Settings

logger = logging.getLogger(__name__)

SPECIAL_ZH = 24884
SPECIAL_SPEECH = 24993
SPECIAL_NEUTRAL = 25004
SPECIAL_NOITN = 25017
VOCAB_SPECIAL_START = 24884


def label_prefix(neutral: bool = True) -> list[int]:
    """Control-token prefix expected by SenseVoice.

    ``neutral=True`` is correct: round 1 filled the emotion slot with 0 (blank)
    and taught the emotion head to predict blank. See docs/training-notes.md.
    """
    return [SPECIAL_ZH, SPECIAL_SPEECH, SPECIAL_NEUTRAL if neutral else 0, SPECIAL_NOITN]


def collapse_ctc(token_ids: Sequence[int]) -> list[int]:
    """CTC greedy collapse: drop repeats, blanks and control tokens."""
    collapsed: list[int] = []
    previous: int | None = None
    for token in token_ids:
        if token != previous:
            if token != 0 and token < VOCAB_SPECIAL_START:
                collapsed.append(int(token))
            previous = token
    return collapsed


class SenseVoiceRecognizer:
    """A SenseVoiceSmall model, optionally with a fine-tuned state dict loaded."""

    def __init__(
        self,
        model_dir: str | Path,
        checkpoint: str | Path | None = None,
        *,
        device: str = "cpu",
        neutral: bool = True,
    ) -> None:
        import torch
        from funasr import AutoModel
        from funasr.frontends.wav_frontend import WavFrontend

        self._torch = torch
        self.model_dir = Path(model_dir)
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.device = device
        self.neutral = neutral

        automodel = AutoModel(model=str(self.model_dir), device="cpu", disable_update=True)
        self.model = automodel.model
        self.tokenizer = automodel.kwargs["tokenizer"]
        self.frontend = WavFrontend(**automodel.kwargs["frontend_conf"])

        if self.checkpoint is not None:
            self._load_checkpoint(self.checkpoint)
        self.model.to(device).eval()

    def _load_checkpoint(self, path: Path) -> None:
        torch = self._torch
        payload: Any
        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception:
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning(
                "checkpoint %s loaded with %d missing and %d unexpected keys",
                path,
                len(missing),
                len(unexpected),
            )
        logger.info("loaded fine-tuned weights from %s", path)

    def list_speakers(self) -> list[str]:  # pragma: no cover - convenience
        return []

    def transcribe(self, source: str | Path | bytes) -> str:
        """Transcribe a file path or raw audio bytes."""
        wav, _ = load_audio(source, target_sr=TARGET_SR)
        return self.transcribe_array(wav)

    def transcribe_array(self, wav: np.ndarray) -> str:
        """Transcribe mono float32 audio already sampled at 16 kHz."""
        torch = self._torch
        samples = np.asarray(wav, dtype=np.float32)
        tensor = torch.from_numpy(samples).unsqueeze(0)
        feats, lengths = self.frontend(tensor, torch.tensor([samples.shape[0]]))
        feats = feats.to(self.device)
        lengths = lengths.to(self.device)
        prefix = torch.tensor([label_prefix(self.neutral)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            encoded, encoded_lengths = self.model.encode(feats, lengths, prefix)
            logits = self.model.ctc.log_softmax(encoded)
            ids = logits[0, : int(encoded_lengths[0])].argmax(dim=-1).tolist()
        return self.tokenizer.decode(collapse_ctc(ids))


def load_recognizer(
    settings: Settings,
    *,
    finetuned: bool = True,
    device: str | None = None,
) -> SenseVoiceRecognizer:
    """Load the fine-tuned or stock recogniser from the configured paths."""
    return SenseVoiceRecognizer(
        settings.asr_base_dir,
        settings.asr_ckpt if finetuned else None,
        device=device or settings.asr_device,
    )
