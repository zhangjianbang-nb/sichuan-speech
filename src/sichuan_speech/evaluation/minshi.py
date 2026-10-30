"""The s/sh (平翘舌) merger test — kept as a documented failure, not as evidence.

Sichuan merges zh/ch/sh into z/c/s, so 「四」(si) and 「十」(shi) should sound nearly
identical while standard Mandarin keeps a spectral-centroid gap. The test measures the
centroid of the fricative at each character onset, using CTC forced alignment against the
known transcript, and reports centroid(s) − centroid(sh).

It fails for two reasons, both reproduced by this module's output: SenseVoice CTC frames
are 56 ms while a fricative lasts 60–100 ms, so the alignment window bleeds into the vowel
(real Mandarin's gap, +67 Hz, came out *smaller* than real Sichuan's, +133 Hz — the opposite
of the linguistic prediction); and the tongue twister makes the SFT model repeat and drop
characters, so its audio is not measurable at all. ``results/minshi_failed.json`` is kept.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..asr import SenseVoiceRecognizer, label_prefix, load_recognizer
from ..audio import TARGET_SR, load_audio, write_wav
from ..config import Settings
from ..tts import CosyVoiceEngine
from .common import TONGUE_TWISTER, load_json, save_json

logger = logging.getLogger(__name__)

S_CHARS = set("四三苏送散僧搜塞丧森")
SH_CHARS = set("十是时说上山水书少收深")


def _centroid(segment: np.ndarray) -> float:
    windowed = segment * np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(windowed, n=2048))
    frequencies = np.fft.rfftfreq(2048, 1 / TARGET_SR)
    band = (frequencies > 1000) & (frequencies < 8000)
    if spectrum[band].sum() < 1e-9:
        return float("nan")
    return float((spectrum[band] * frequencies[band]).sum() / spectrum[band].sum())


def _analyse(recognizer: SenseVoiceRecognizer, path: str, text: str) -> dict | None:
    import torch

    wav, _ = load_audio(path, target_sr=TARGET_SR)
    duration = len(wav) / TARGET_SR
    feats, lengths = recognizer.frontend(
        torch.from_numpy(wav).unsqueeze(0), torch.tensor([len(wav)])
    )
    feats = feats.to(recognizer.device)
    lengths = lengths.to(recognizer.device)
    ids = recognizer.tokenizer.encode(text)
    prefix = torch.tensor([label_prefix()], dtype=torch.long, device=recognizer.device)
    with torch.no_grad():
        encoded, encoded_lengths = recognizer.model.encode(feats, lengths, prefix)
        logits = recognizer.model.ctc.log_softmax(encoded)
    frames = int(encoded_lengths[0])
    hop = duration / frames
    log_probs = logits[0, :frames].unsqueeze(0).float()
    targets = torch.tensor([ids], dtype=torch.long)
    try:
        import torchaudio.functional as F

        alignment, _ = F.forced_align(
            log_probs, targets, torch.tensor([frames]), torch.tensor([len(ids)]), blank=0
        )
    except Exception as error:
        logger.warning("forced_align unavailable for %s: %s", path, error)
        return None

    # forced_align returns label ids per frame, not target positions. Group frames into
    # runs and walk runs and targets in lockstep so repeated characters get distinct spans.
    runs: list[list[int]] = []
    previous: int | None = None
    for frame, label in enumerate(alignment[0].tolist()):
        if label == 0:
            previous = None
            continue
        if previous is None or label != previous:
            runs.append([label, frame, frame])
            previous = label
        else:
            runs[-1][2] = frame

    spans: dict[int, tuple[int, int]] = {}
    pointer = 0
    for label, start, end in runs:
        if pointer < len(ids) and ids[pointer] == label:
            spans[pointer] = (start, end)
            pointer += 1

    s_values: list[float] = []
    sh_values: list[float] = []
    for index, char in enumerate(text):
        span = spans.get(index)
        if not span:
            continue
        start, end = span
        window = min(0.085, max(1, end - start + 1) * hop * 0.45)
        begin_sample = int(start * TARGET_SR * hop)
        finish_sample = begin_sample + int(window * TARGET_SR)
        if (
            finish_sample > len(wav)
            or begin_sample >= len(wav)
            or finish_sample - begin_sample < 160
        ):
            continue
        value = _centroid(wav[begin_sample:finish_sample])
        if np.isnan(value):
            continue
        if char in S_CHARS:
            s_values.append(value)
        elif char in SH_CHARS:
            sh_values.append(value)
    return {"s": s_values, "sh": sh_values, "frames": frames, "hop_ms": round(hop * 1000, 1)}


def _collect(
    recognizer: SenseVoiceRecognizer, items: list[tuple[str, str]], name: str
) -> dict | None:
    s_all: list[float] = []
    sh_all: list[float] = []
    used = 0
    for path, text in items:
        result = _analyse(recognizer, path, text)
        if not result or not result["s"] or not result["sh"]:
            continue
        used += 1
        s_all += result["s"]
        sh_all += result["sh"]
    if not s_all or not sh_all:
        logger.info("%-18s no measurable data", name)
        return None
    mean_s, mean_sh = float(np.mean(s_all)), float(np.mean(sh_all))
    logger.info(
        "%-18s utts %2d  n_s %3d n_sh %3d  centroid_s %.0f  centroid_sh %.0f  gap %+.0f Hz",
        name,
        used,
        len(s_all),
        len(sh_all),
        mean_s,
        mean_sh,
        mean_s - mean_sh,
    )
    return {
        "utts": used,
        "n_s": len(s_all),
        "n_sh": len(sh_all),
        "centroid_s": round(mean_s, 1),
        "centroid_sh": round(mean_sh, 1),
        "gap_hz": round(mean_s - mean_sh, 1),
    }


def _pick(rows: list[dict], per_set: int) -> list[tuple[str, str]]:
    picked = []
    for row in rows:
        text = row["text"]
        if (
            any(char in S_CHARS for char in text)
            and any(char in SH_CHARS for char in text)
            and 1.0 < len(text) < 30
        ):
            picked.append((row["wav"], text))
        if len(picked) >= per_set:
            break
    return picked


def evaluate_minshi(
    settings: Settings,
    *,
    sft_dir: str | Path | None = None,
    out: str | Path | None = None,
    per_set: int = 25,
    n_takes: int = 3,
) -> dict:
    """Write ``minshi.json``; every number here documents why the test is unusable."""
    import torch

    output = Path(out) if out else settings.eval_root / "final" / "minshi"
    output.mkdir(parents=True, exist_ok=True)
    sft_model_dir = Path(sft_dir) if sft_dir else settings.sft_model_dir
    recognizer = load_recognizer(settings, finetuned=True)

    report: dict[str, Any] = {}
    report["real_sichuan"] = _collect(
        recognizer,
        _pick(load_json(settings.data_root / "eval" / "kespeech_sw.json"), per_set),
        "real_sichuan",
    )
    report["real_mandarin"] = _collect(
        recognizer,
        _pick(load_json(settings.data_root / "eval" / "kespeech_mandarin.json"), per_set),
        "real_mandarin",
    )

    tongue_dir = output / "tongue"
    tongue_dir.mkdir(parents=True, exist_ok=True)
    engines = [
        ("tts_sft", sft_model_dir, "sichuan_female", None),
        (
            "tts_base",
            settings.base_model_dir,
            None,
            (settings.official_prompt_text, str(settings.official_prompt_wav)),
        ),
    ]
    for tag, model_dir, spk, prompt in engines:
        engine = CosyVoiceEngine(model_dir, fp16=settings.tts_fp16, settings=settings)
        for take in range(n_takes):
            torch.manual_seed(2000 + take)
            np.random.seed(2000 + take)
            if spk:
                stream = engine.model.inference_sft(TONGUE_TWISTER, spk, stream=False)
            else:
                assert prompt is not None
                stream = engine.model.inference_zero_shot(
                    TONGUE_TWISTER, prompt[0], prompt[1], stream=False
                )
            wav = np.concatenate([chunk["tts_speech"].squeeze().cpu().numpy() for chunk in stream])
            path = tongue_dir / f"{tag}_{take}.wav"
            write_wav(path, wav, engine.sample_rate)
            logger.info("generated %s (%.2f s)", path, len(wav) / engine.sample_rate)
        del engine
        torch.cuda.empty_cache()

    for tag in ("tts_sft", "tts_base"):
        files = sorted(glob.glob(str(tongue_dir / f"{tag}_*.wav")))
        report[tag] = _collect(recognizer, [(path, TONGUE_TWISTER) for path in files], tag)

    save_json(output / "minshi.json", report)
    return report
