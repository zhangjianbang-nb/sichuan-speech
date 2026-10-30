"""ASR evaluation: stock SenseVoiceSmall vs Sichuan checkpoints on held-out sets.

The KeSpeech Southwestern and Mandarin sets come from the same corpus and
recording channel, so the CER gap between them is a clean dialect signal, and the
Mandarin set is the forgetting guard. Both sides must use ``use_itn=False``:
mixing ITN settings produced a fake improvement in round 1 (see
``docs/training-notes.md``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from ..asr import SenseVoiceRecognizer, load_recognizer
from ..config import Settings
from ..text import char_error_stats, strip_special_tokens
from .common import load_json, save_json

logger = logging.getLogger(__name__)


def _score(
    rows: Sequence[dict],
    transcribe: Callable[[str], str],
    name: str,
) -> dict:
    total_errors = total_chars = 0
    errors: list[dict] = []
    for row in rows:
        hypothesis = transcribe(str(row["wav"]))
        stats = char_error_stats(str(row["text"]), hypothesis)
        total_errors += stats.errors
        total_chars += stats.ref_chars
        if stats.errors:
            errors.append(
                {
                    "ref": row["text"],
                    "hyp": hypothesis,
                    "cer": round(stats.cer, 3),
                }
            )
    mean = round(total_errors / max(1, total_chars), 4)
    logger.info("%-34s CER %.4f  (%d utts)", name, mean, len(rows))
    return {"mean_cer": mean, "n": len(rows), "errors": errors[:25]}


def _base_transcriber(settings: Settings, use_itn: bool) -> Callable[[str], str]:
    if use_itn:
        from funasr import AutoModel

        model = AutoModel(model=str(settings.asr_base_dir), device="cpu", disable_update=True)

        def transcribe(path: str) -> str:
            result = model.generate(
                input=path, cache={}, language="zh", use_itn=True, batch_size_s=60
            )
            return strip_special_tokens(result[0]["text"])

        return transcribe

    recognizer = load_recognizer(settings, finetuned=False)
    return recognizer.transcribe


def evaluate_asr(
    settings: Settings,
    *,
    ckpts: Sequence[str] = (),
    out: str | Path | None = None,
    use_itn: bool = False,
    device: str = "cpu",
) -> dict:
    """Score the base model and every checkpoint in ``ckpts``."""
    eval_root = settings.data_root / "eval"
    southwestern = load_json(eval_root / "kespeech_sw.json")
    mandarin = load_json(eval_root / "kespeech_mandarin.json")
    report: dict = {"base": {}, "ft": {}, "use_itn": use_itn}

    base_transcribe = _base_transcriber(settings, use_itn)
    report["base"]["sw"] = _score(southwestern, base_transcribe, "base / kespeech_southwestern")
    report["base"]["man"] = _score(mandarin, base_transcribe, "base / kespeech_mandarin")

    for checkpoint in ckpts:
        recognizer = SenseVoiceRecognizer(settings.asr_base_dir, checkpoint, device=device)
        tag = Path(checkpoint).name
        report["ft"][tag] = {
            "sw": _score(southwestern, recognizer.transcribe, f"{tag} / southwestern"),
            "man": _score(mandarin, recognizer.transcribe, f"{tag} / mandarin"),
        }

    target = Path(out) if out else settings.eval_root / "asr_report.json"
    save_json(target, report)
    logger.info("wrote %s", target)
    return report
