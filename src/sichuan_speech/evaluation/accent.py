"""Accent calibration: how Sichuan-flavoured is a rendering, relative to real speech.

Two signals, both anchored on human recordings rather than on an assumption:

* acoustic statistics (F0 register, range, contour direction, speaking rate);
* an optional KeSpeech-trained Southwestern-vs-Mandarin classifier.

The classifier is recorded but not used as evidence: speaker-disjoint cross-validation
accuracy was 0.567, i.e. barely above chance. The acoustic table is what the report cites.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Settings
from .common import SICHUAN_SENTENCES, acoustic_features, aggregate, generate, load_json, save_json

logger = logging.getLogger(__name__)

KEYS = ("f0_med_semi", "f0_range", "f0_std", "fall_frac", "rate_syl_s", "dur")


def _score_classifier(paths: list[str], checkpoint: Path) -> float | None:
    if not checkpoint.exists():
        return None
    import torch

    from ..training.dialect_clf import _librosa_mel, _net_class

    net = _net_class()
    net.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
    net.eval()
    probabilities = []
    with torch.no_grad():
        for path in paths:
            tensor = torch.from_numpy(_librosa_mel(path)).unsqueeze(0).unsqueeze(0)
            probabilities.append(float(torch.softmax(net(tensor), -1)[0, 1]))
    return round(float(np.mean(probabilities)), 3) if probabilities else None


def _acoustic(paths: list[str]) -> dict:
    return aggregate([acoustic_features(path) for path in paths])


def evaluate_accent(
    settings: Settings,
    *,
    sft_dir: str | Path | None = None,
    out: str | Path | None = None,
    clf: str | Path | None = None,
    n_sent: int = 10,
    n_anchor: int = 40,
) -> dict:
    """Write ``accent_report.json`` with the anchor bands and the four synthesised sets."""
    import torch

    from ..tts import CosyVoiceEngine

    output = Path(out) if out else settings.eval_root / "accent"
    output.mkdir(parents=True, exist_ok=True)
    classifier = Path(clf) if clf else settings.work_root / "clf_sw.pt"
    sft_model_dir = Path(sft_dir) if sft_dir else settings.sft_model_dir
    sentences = SICHUAN_SENTENCES[:n_sent]

    sichuan_rows = load_json(settings.data_root / "sft" / "all_sichuan.json")[:n_anchor]
    mandarin_rows = load_json(settings.data_root / "eval" / "kespeech_mandarin.json")[:n_anchor]
    sichuan_paths = [row[1] for row in sichuan_rows]
    mandarin_paths = [row["wav"] for row in mandarin_rows]

    report: dict[str, Any] = {
        "anchor_real_sichuan": _acoustic(sichuan_paths),
        "anchor_real_mandarin": _acoustic(mandarin_paths),
    }
    report["anchor_real_sichuan"]["p_sw"] = _score_classifier(sichuan_paths, classifier)
    report["anchor_real_mandarin"]["p_sw"] = _score_classifier(mandarin_paths, classifier)

    mandarin_prompt = (settings.official_prompt_text, str(settings.official_prompt_wav))
    sichuan_prompt = (settings.prompt_text, str(settings.prompt_wav))

    base_engine = CosyVoiceEngine(settings.base_model_dir, fp16=settings.tts_fp16, settings=settings)
    base_mandarin = generate(
        base_engine.model, sentences, output / "base_mandarin_prompt", mode="zero_shot", prompt=mandarin_prompt
    )
    base_sichuan = generate(
        base_engine.model, sentences, output / "base_sichuan_prompt", mode="zero_shot", prompt=sichuan_prompt
    )
    del base_engine
    torch.cuda.empty_cache()

    sft_engine = CosyVoiceEngine(sft_model_dir, fp16=settings.tts_fp16, settings=settings)
    sft = generate(sft_engine.model, sentences, output / "sft", spk="sichuan_female")
    del sft_engine

    sets = {
        "base_mandarin_prompt": [str(path) for chunk in base_mandarin for path in chunk],
        "base_sichuan_prompt": [str(path) for chunk in base_sichuan for path in chunk],
        "sft": [str(path) for chunk in sft for path in chunk],
    }
    for name, paths in sets.items():
        report[name] = _acoustic(paths)
        report[name]["p_sw"] = _score_classifier(paths, classifier)

    save_json(output / "accent_report.json", report)
    header = "%-24s %s" % ("set", "  ".join("%10s" % key for key in KEYS))
    logger.info(header)
    for name, values in report.items():
        logger.info("%-24s %s", name, "  ".join("%10s" % values.get(key) for key in KEYS))
    return report
