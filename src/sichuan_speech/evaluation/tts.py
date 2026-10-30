"""TTS evaluation: intelligibility (ASR CER), accent calibration and checkpoint sweep.

Sets compared, all from the same vocoder so any synthetic-domain bias cancels:

* ``base_mandarin_prompt``  stock CosyVoice2 + official Mandarin prompt
* ``base_sichuan_prompt``   stock CosyVoice2 + Sichuan prompt (accent cloned)
* ``sft_female``            Sichuan-SFT LLM + MagicHub Chengdu female embedding
* ``sft_avg``               Sichuan-SFT LLM + mean KeSpeech speaker embedding
* ``sft_mandarin``          regression guard: Sichuan LLM reading Mandarin text

Anchors (40 real Sichuan, 40 real Mandarin) give reference ranges for the
acoustic statistics. ``cross_asr`` is kept as an auxiliary observation only: its
effect is too small to separate "dialect adaptation" from "the Sichuan recogniser
is simply better on everything".
"""

from __future__ import annotations

import glob
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..asr import load_recognizer
from ..config import Settings
from ..text import char_error_stats
from .common import (
    MANDARIN_SENTENCES,
    SICHUAN_SENTENCES,
    acoustic_features,
    aggregate,
    generate,
    load_json,
    save_json,
)

logger = logging.getLogger(__name__)


def _score_set(
    paths_per_sentence: Sequence[Sequence[Path]],
    references: Sequence[str],
    name: str,
    transcribe,
) -> dict:
    total_errors = total_chars = 0
    rows = []
    for index, paths in enumerate(paths_per_sentence):
        for path in paths:
            hypothesis = transcribe(str(path))
            stats = char_error_stats(references[index], hypothesis)
            total_errors += stats.errors
            total_chars += stats.ref_chars
            rows.append(
                {
                    "wav": str(path),
                    "ref": references[index],
                    "hyp": hypothesis,
                    "cer": round(stats.cer, 4),
                }
            )
    flat = [path for paths in paths_per_sentence for path in paths]
    result: dict[str, Any] = {
        "mean_cer": round(total_errors / max(1, total_chars), 4),
        "acoustic": aggregate([acoustic_features(path) for path in flat]),
        "rows": rows,
    }
    acoustic = result["acoustic"]
    logger.info(
        "%-22s CER %.4f  f0_med %s  rate %s  fall %s",
        name,
        result["mean_cer"],
        acoustic.get("f0_med_semi"),
        acoustic.get("rate_syl_s"),
        acoustic.get("fall_frac"),
    )
    return result


def _prompts(settings: Settings) -> tuple[tuple[str, str], tuple[str, str]]:
    sichuan = (settings.prompt_text, str(settings.prompt_wav))
    mandarin = (settings.official_prompt_text, str(settings.official_prompt_wav))
    return sichuan, mandarin


def evaluate_tts(
    settings: Settings,
    *,
    sft_dir: str | Path | None = None,
    out: str | Path | None = None,
    n_anchor: int = 40,
) -> dict:
    """Full TTS evaluation; writes ``tts_eval.json``."""
    import torch

    from ..tts import CosyVoiceEngine

    output = Path(out) if out else settings.eval_root / "final"
    output.mkdir(parents=True, exist_ok=True)
    sft_model_dir = Path(sft_dir) if sft_dir else settings.sft_model_dir
    sichuan_prompt, mandarin_prompt = _prompts(settings)

    base_asr = load_recognizer(settings, finetuned=False)
    transcribe = base_asr.transcribe

    sichuan_rows = load_json(settings.data_root / "sft" / "all_sichuan.json")[:n_anchor]
    mandarin_rows = load_json(settings.data_root / "eval" / "kespeech_mandarin.json")[:n_anchor]

    report: dict[str, Any] = {}
    report["anchor_real_sichuan"] = {
        "acoustic": aggregate([acoustic_features(row[1]) for row in sichuan_rows]),
        "mean_cer": round(
            sum(char_error_stats(row[2], transcribe(row[1])).errors for row in sichuan_rows)
            / max(1, sum(len(row[2]) for row in sichuan_rows)),
            4,
        ),
    }
    report["anchor_real_mandarin"] = {
        "acoustic": aggregate([acoustic_features(row["wav"]) for row in mandarin_rows]),
        "mean_cer": round(
            sum(
                char_error_stats(row["text"], transcribe(row["wav"])).errors
                for row in mandarin_rows
            )
            / max(1, sum(len(row["text"]) for row in mandarin_rows)),
            4,
        ),
    }
    logger.info("anchors done")

    base_engine = CosyVoiceEngine(
        settings.base_model_dir, fp16=settings.tts_fp16, settings=settings
    )
    base_mandarin = generate(
        base_engine.model,
        SICHUAN_SENTENCES,
        output / "base_mandarin_prompt",
        mode="zero_shot",
        prompt=mandarin_prompt,
    )
    base_sichuan = generate(
        base_engine.model,
        SICHUAN_SENTENCES,
        output / "base_sichuan_prompt",
        mode="zero_shot",
        prompt=sichuan_prompt,
    )
    del base_engine
    torch.cuda.empty_cache()

    sft_engine = CosyVoiceEngine(sft_model_dir, fp16=settings.tts_fp16, settings=settings)
    speakers = sft_engine.list_speakers()
    logger.info("sft speakers: %s", speakers)
    sft_female = generate(
        sft_engine.model, SICHUAN_SENTENCES, output / "sft_female", spk="sichuan_female"
    )
    sft_avg = (
        generate(sft_engine.model, SICHUAN_SENTENCES, output / "sft_avg", spk="sichuan_avg")
        if "sichuan_avg" in speakers
        else None
    )
    sft_mandarin = generate(
        sft_engine.model, MANDARIN_SENTENCES, output / "sft_mandarin", spk="sichuan_female"
    )
    del sft_engine

    report["base_mandarin_prompt"] = _score_set(
        base_mandarin, SICHUAN_SENTENCES, "base_mandarin_prompt", transcribe
    )
    report["base_sichuan_prompt"] = _score_set(
        base_sichuan, SICHUAN_SENTENCES, "base_sichuan_prompt", transcribe
    )
    report["sft_female"] = _score_set(sft_female, SICHUAN_SENTENCES, "sft_female", transcribe)
    if sft_avg is not None:
        report["sft_avg"] = _score_set(sft_avg, SICHUAN_SENTENCES, "sft_avg", transcribe)
    report["sft_mandarin_regression"] = _score_set(
        sft_mandarin, MANDARIN_SENTENCES, "sft_mandarin_regression", transcribe
    )

    save_json(output / "tts_eval.json", report)
    logger.info("wrote %s", output / "tts_eval.json")
    return report


def cross_asr(
    settings: Settings,
    *,
    out: str | Path | None = None,
    ft_ckpt: str | Path | None = None,
    eval_dir: str | Path | None = None,
) -> dict:
    """Two recognisers score the same synthesised audio (auxiliary observation)."""
    from ..asr import SenseVoiceRecognizer

    output = Path(out) if out else settings.eval_root / "final"
    directory = Path(eval_dir) if eval_dir else output
    checkpoint = Path(ft_ckpt) if ft_ckpt else settings.asr_ckpt
    base = load_recognizer(settings, finetuned=False)
    finetuned = SenseVoiceRecognizer(settings.asr_base_dir, checkpoint, device=settings.asr_device)

    subsets = [
        name
        for name in (
            "base_mandarin_prompt",
            "base_sichuan_prompt",
            "sft_female",
            "sft_avg",
            "sft_mandarin",
        )
        if (directory / name).is_dir()
    ]
    report: dict[str, Any] = {}
    for subset in subsets:
        references = MANDARIN_SENTENCES if subset == "sft_mandarin" else SICHUAN_SENTENCES
        files = sorted(glob.glob(str(directory / subset / "*.wav")))
        base_errors = base_chars = ft_errors = ft_chars = 0
        rows = []
        for path in files:
            index = int(Path(path).name.split("_")[0])
            reference = references[index]
            hyp_base = base.transcribe(path)
            hyp_ft = finetuned.transcribe(path)
            stats_base = char_error_stats(reference, hyp_base)
            stats_ft = char_error_stats(reference, hyp_ft)
            base_errors += stats_base.errors
            base_chars += stats_base.ref_chars
            ft_errors += stats_ft.errors
            ft_chars += stats_ft.ref_chars
            rows.append(
                {
                    "wav": Path(path).name,
                    "ref": reference,
                    "base": hyp_base,
                    "ft": hyp_ft,
                    "cer_base": round(stats_base.cer, 4),
                    "cer_ft": round(stats_ft.cer, 4),
                }
            )
        cer_base = round(base_errors / max(1, base_chars), 4)
        cer_ft = round(ft_errors / max(1, ft_chars), 4)
        report[subset] = {
            "n": len(files),
            "cer_base": cer_base,
            "cer_ft": cer_ft,
            "delta": round(cer_ft - cer_base, 4),
            "rows": rows,
        }
        logger.info(
            "%-22s base %.4f  ft %.4f  delta %+.4f", subset, cer_base, cer_ft, cer_ft - cer_base
        )

    save_json(output / "cross_asr.json", report)
    return report


def sweep_checkpoints(
    settings: Settings,
    *,
    model_dir: str | Path | None = None,
    ckpt_dir: str | Path | None = None,
    out: str | Path | None = None,
    spk: str = "sichuan_female",
    n_sent: int = 20,
) -> dict:
    """Score every checkpoint on the accent objective, not on CV loss.

    CV loss measures token prediction, not accent strength. This sweep generates
    the same sentences under fixed seeds and reports, per checkpoint: stock-ASR
    CER (rises as pronunciation drifts from Mandarin), Sichuan-ASR CER (stays low
    only if the audio is intelligible) and syllables/second (real Sichuan anchor:
    4.67). The reported numbers are in ``results/tts_epoch_sweep.json``.
    """
    import torch

    from ..tts import CosyVoiceEngine

    model_directory = Path(model_dir) if model_dir else settings.sft_model_dir
    checkpoint_directory = Path(ckpt_dir) if ckpt_dir else settings.work_root / "exp" / "llm2"
    output = Path(out) if out else settings.eval_root / "sweep"
    output.mkdir(parents=True, exist_ok=True)
    sentences = SICHUAN_SENTENCES[:n_sent]

    base = load_recognizer(settings, finetuned=False)
    finetuned = load_recognizer(settings, finetuned=True)
    engine = CosyVoiceEngine(model_directory, fp16=settings.tts_fp16, settings=settings)
    logger.info("speakers: %s", engine.list_speakers())

    checkpoints = sorted(
        glob.glob(str(checkpoint_directory / "epoch_*_whole.pt")),
        key=lambda path: int(re.search(r"epoch_(\d+)", path).group(1)),
    )
    candidates: list[tuple[str, str | None]] = [("base", None)] + [
        (Path(path).name.replace("_whole.pt", ""), path) for path in checkpoints
    ]

    report: dict[str, Any] = {}
    for tag, checkpoint in candidates:
        source = checkpoint or str(settings.base_model_dir / "llm.pt")
        state = torch.load(source, map_location="cpu", weights_only=True)
        state.pop("epoch", None)
        state.pop("step", None)
        engine.model.model.llm.load_state_dict(state, strict=True)
        engine.model.model.llm.eval()
        directory = output / tag
        directory.mkdir(parents=True, exist_ok=True)
        base_errors = base_chars = ft_errors = ft_chars = 0
        acoustics = []
        for index, text in enumerate(sentences):
            torch.manual_seed(1000 + index)
            np.random.seed(1000 + index)
            chunks = [
                output_chunk["tts_speech"].squeeze().cpu().numpy()
                for output_chunk in engine.model.inference_sft(text, spk, stream=False)
            ]
            from ..audio import write_wav

            path = write_wav(
                directory / f"{index:02d}.wav", np.concatenate(chunks), engine.sample_rate
            )
            stats_base = char_error_stats(text, base.transcribe(str(path)))
            stats_ft = char_error_stats(text, finetuned.transcribe(str(path)))
            base_errors += stats_base.errors
            base_chars += stats_base.ref_chars
            ft_errors += stats_ft.errors
            ft_chars += stats_ft.ref_chars
            acoustics.append(acoustic_features(path))
        cer_base = round(base_errors / max(1, base_chars), 4)
        cer_ft = round(ft_errors / max(1, ft_chars), 4)
        aggregated = aggregate(acoustics)
        report[tag] = {
            "cer_base": cer_base,
            "cer_ft": cer_ft,
            "rate": aggregated.get("rate_syl_s"),
            "dur": aggregated.get("dur"),
        }
        logger.info(
            "%-8s cer_base %.4f  cer_ft %.4f  rate %.2f  dur %.2f",
            tag,
            cer_base,
            cer_ft,
            report[tag]["rate"] or 0,
            report[tag]["dur"] or 0,
        )
        save_json(output / "sweep.json", report)

    return report
