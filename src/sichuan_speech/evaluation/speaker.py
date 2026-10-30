"""Speaker fidelity: does the audio carry the requested timbre?

CosyVoice2's flow matching stage is the only stage that consumes the speaker embedding —
the LLM does not (see docs/training-notes.md). So cosine similarity between the generated
audio's campplus embedding and the embedding used at synthesis measures whether the flow
honoured the condition. ``sichuan_female`` scores 0.678; ``sichuan_avg`` scores 0.369,
which is expected: an averaged embedding does not correspond to any real speaker.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np

from ..audio import TARGET_SR, load_audio
from ..config import Settings
from .common import save_json

logger = logging.getLogger(__name__)


def _embedding(session, path: str | Path) -> np.ndarray:
    import torch
    import torchaudio.compliance.kaldi as kaldi

    wav, _ = load_audio(path, target_sr=TARGET_SR)
    tensor = torch.from_numpy(wav).unsqueeze(0)
    fbank = kaldi.fbank(tensor, num_mel_bins=80, sample_frequency=TARGET_SR, dither=0.0)
    fbank = fbank - fbank.mean(0, keepdim=True)
    output = session.run(None, {session.get_inputs()[0].name: fbank.unsqueeze(0).numpy()})[0]
    vector = torch.from_numpy(output).reshape(1, -1)
    return torch.nn.functional.normalize(vector, dim=1).numpy()


def check_speaker_similarity(
    settings: Settings,
    *,
    out: str | Path | None = None,
    eval_dir: str | Path | None = None,
    sft_dir: str | Path | None = None,
    subsets: tuple[tuple[str, str], ...] = (("sft_female", "sichuan_female"), ("sft_avg", "sichuan_avg")),
) -> dict:
    """Write ``speaker_sim.json`` with mean and minimum cosine per synthesised subset."""
    import onnxruntime
    import torch

    directory = Path(eval_dir) if eval_dir else settings.eval_root / "final"
    output = Path(out) if out else directory
    sft_model_dir = Path(sft_dir) if sft_dir else settings.sft_model_dir

    options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(
        str(settings.base_model_dir / "campplus.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    spk2info = torch.load(str(sft_model_dir / "spk2info.pt"), map_location="cpu", weights_only=False)

    report: dict[str, dict] = {}
    for subset, speaker in subsets:
        paths = sorted(glob.glob(str(directory / subset / "*.wav")))
        if not paths:
            continue
        target = torch.nn.functional.normalize(
            spk2info[speaker]["embedding"].reshape(1, -1), dim=1
        ).numpy()
        similarities = []
        for path in paths:
            try:
                similarities.append(float((_embedding(session, path) * target).sum()))
            except Exception as error:  # noqa: BLE001 - one bad file must not abort the report
                logger.warning("skipped %s: %s", path, error)
        if not similarities:
            continue
        report[subset] = {
            "n": len(similarities),
            "mean_cos": round(float(np.mean(similarities)), 3),
            "min_cos": round(float(np.min(similarities)), 3),
        }
        logger.info("%s %s", subset, report[subset])

    prompt_wav = settings.prompt_wav
    if prompt_wav.exists():
        target = torch.nn.functional.normalize(
            spk2info["sichuan_female"]["embedding"].reshape(1, -1), dim=1
        ).numpy()
        report["prompt_self"] = {"cos": round(float((_embedding(session, prompt_wav) * target).sum()), 3)}
        logger.info("prompt_self %s", report["prompt_self"])

    save_json(output / "speaker_sim.json", report)
    return report
