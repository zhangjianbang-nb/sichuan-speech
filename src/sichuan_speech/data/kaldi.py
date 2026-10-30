"""Kaldi-style manifest helpers.

CosyVoice2 consumes ``wav.scp``/``text``/``utt2spk``/``spk2utt``. Rows are
5-tuples ``[utt_id, wav_path, text, speaker, duration_seconds]``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

Row = Sequence[object]


def _normalize_utt(utt: str) -> str:
    return str(utt).replace(" ", "_")


def write_kaldi_dir(directory: str | Path, rows: Iterable[Row]) -> dict:
    """Write ``wav.scp``, ``text``, ``utt2spk`` and ``spk2utt`` for ``rows``."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    rows = [list(row) for row in rows]

    spk2utt: dict[str, list[str]] = {}
    with (
        (target / "wav.scp").open("w", encoding="utf-8") as wav_scp,
        (target / "text").open("w", encoding="utf-8") as text,
        (target / "utt2spk").open("w", encoding="utf-8") as utt2spk,
    ):
        for row in rows:
            utt, path, transcript, speaker = (
                _normalize_utt(str(row[0])),
                str(row[1]),
                str(row[2]),
                str(row[3]),
            )
            wav_scp.write(f"{utt} {path}\n")
            text.write(f"{utt} {transcript}\n")
            utt2spk.write(f"{utt} {speaker}\n")
            spk2utt.setdefault(speaker, []).append(utt)

    with (target / "spk2utt").open("w", encoding="utf-8") as handle:
        for speaker, utts in spk2utt.items():
            handle.write(f"{speaker} {' '.join(utts)}\n")

    hours = sum(float(str(row[4])) for row in rows) / 3600.0 if rows else 0.0
    logger.info(
        "wrote %s: %d utts, %d speakers, %.2f h", target, len(rows), len(spk2utt), hours
    )
    return {"utts": len(rows), "speakers": len(spk2utt), "hours": round(hours, 2)}


def write_manifest(path: str | Path, rows: Iterable[Row]) -> Path:
    """Write the JSON manifest used by the training scripts."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([list(row) for row in rows], ensure_ascii=False), encoding="utf-8"
    )
    return target


def read_manifest(path: str | Path) -> list[list]:
    """Read a JSON manifest."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
