"""Shared helpers for the evaluation scripts."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SICHUAN_SENTENCES: list[str] = [
    "今天天气好安逸哦，我们出去耍一哈嘛。",
    "你吃饭没有得？没得的话我们一起切吃火锅。",
    "这个事情我晓得，你莫着急，慢慢来。",
    "老板儿，来二两小面，多放点海椒。",
    "你娃儿最近在忙啥子嘛？",
    "昨晚上那个雨下得才大哦，裤子都打湿完了。",
    "这个菜巴适得板，再给我舀一碗饭。",
    "莫慌，等我把这个事情搞归一了再说。",
    "他屋头那个猫儿又跑到房顶上去了。",
    "我们是四川人，说话就是这个味道。",
    "你莫要扯把子，这个事情我清楚得很。",
    "走嘛，我们切河边头转一圈，顺便买点卤菜。",
    "那个妹儿长得乖得很，说话也温柔。",
    "天都黑了，你还不回来吃饭啊？",
    "我跟你讲，这个事情办得巴适。",
    "你不要扭到我费，我自己晓得咋个办。",
    "来嘛，坐下喝口茶，慢慢摆龙门阵。",
    "这个瓜娃子，一天到黑就晓得耍。",
    "我屋头的老汉儿今年七十多了，身体还硬朗。",
    "四川的火锅嘛，就是要吃辣才安逸。",
]

MANDARIN_SENTENCES: list[str] = [
    "今天天气很好，我们出去走一走吧。",
    "你吃饭了吗？没有的话我们一起去吃火锅。",
    "这件事情我知道，你不要着急，慢慢来。",
    "老板，来二两小面，多放一点辣椒。",
]

TONGUE_TWISTER = "四是四，十是十，十四是十四，四十是四十。"


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


def acoustic_features(path: str | Path) -> dict[str, float] | None:
    """F0 register/range, falling-contour fraction and speaking rate.

    Returns ``None`` when the clip is too short or unvoiced. These numbers are
    descriptive, not a dialect test: F0 cannot be compared across speakers.
    """
    import librosa

    try:
        wav, sr = librosa.load(str(path), sr=16000)
        wav, _ = librosa.effects.trim(wav, top_db=30)
        if len(wav) < sr * 0.3:
            return None
        f0, _, _ = librosa.pyin(wav, fmin=70, fmax=450, sr=sr, frame_length=1024, hop_length=160)
        voiced = f0[~np.isnan(f0)]
        if len(voiced) < 10:
            return None
        semitones = 12 * np.log2(voiced / 100.0)
        deltas = np.diff(semitones)
        deltas = deltas[np.abs(deltas) > 0.02]
        envelope = librosa.onset.onset_strength(y=wav, sr=sr, hop_length=160)
        onsets = librosa.onset.onset_detect(
            onset_envelope=envelope, sr=sr, hop_length=160, units="time"
        )
        return {
            "dur": round(float(len(wav) / sr), 2),
            "f0_med_semi": round(float(np.median(semitones)), 2),
            "f0_range": round(
                float(np.percentile(semitones, 90) - np.percentile(semitones, 10)), 2
            ),
            "f0_std": round(float(np.std(semitones)), 2),
            "fall_frac": round(float((deltas < 0).mean()), 3),
            "rate_syl_s": round(float(len(onsets) / max(1e-3, len(wav) / sr)), 2),
        }
    except Exception as exc:
        logger.debug("acoustic features failed for %s: %s", path, exc)
        return None


def aggregate(rows: Sequence[dict[str, float] | None]) -> dict[str, Any]:
    """Mean of each feature across rows, skipping ``None``."""
    valid = [row for row in rows if row]
    if not valid:
        return {}
    out = {key: round(float(np.mean([row[key] for row in valid])), 3) for key in valid[0]}
    out["n"] = len(valid)
    return out


def generate(
    model: Any,
    sentences: Sequence[str],
    out_dir: str | Path,
    *,
    mode: str = "sft",
    spk: str | None = None,
    prompt: tuple[str, str] | None = None,
    seed_base: int | None = None,
) -> list[list[Path]]:
    """Render sentences with a loaded CosyVoice model; returns per-sentence chunk paths."""
    from ..audio import write_wav

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results: list[list[Path]] = []
    for index, text in enumerate(sentences):
        if seed_base is not None:
            import torch

            torch.manual_seed(seed_base + index)
            np.random.seed(seed_base + index)
        if mode == "zero_shot":
            if prompt is None:
                raise ValueError("zero_shot generation needs (prompt_text, prompt_wav)")
            iterator = model.inference_zero_shot(text, prompt[0], prompt[1], stream=False)
        else:
            iterator = model.inference_sft(text, spk, stream=False)
        paths = []
        for chunk_index, output in enumerate(iterator):
            path = directory / f"{index:02d}_{chunk_index}.wav"
            write_wav(path, output["tts_speech"].squeeze().cpu().numpy(), model.sample_rate)
            paths.append(path)
        results.append(paths)
    return results
