"""Extract the KeSpeech evaluation sets from the Hugging Face parquet shards.

KeSpeech records Southwestern Mandarin and Mandarin in the same sessions, so the
CER gap between the two sets is a clean dialect signal and the Mandarin set is a
forgetting guard. Official train split is application-only; the public release is
the test split, so training uses one part of it and evaluation uses a disjoint
part (excluded by utterance id in :mod:`sichuan_speech.data.manifest`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import Settings

logger = logging.getLogger(__name__)


def extract_eval_sets(
    settings: Settings,
    *,
    sw_limit: int = 400,
    man_limit: int = 300,
) -> dict:
    """Write ``data/eval/kespeech_sw.json`` and ``kespeech_mandarin.json``."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read KeSpeech parquet files") from exc

    source = settings.data_root / "raw" / "kespeech"
    output = settings.data_root / "eval"
    output.mkdir(parents=True, exist_ok=True)
    files = _parquet_files(source)
    if not files:
        raise FileNotFoundError(f"no KeSpeech parquet files under {source}")

    southwestern: list[tuple[str, bytes, str]] = []
    mandarin: list[tuple[str, bytes, str]] = []
    seen: set[str] = set()
    for parquet in files:
        for record in pq.read_table(parquet).to_pylist():
            uid = record.get("ID") or ""
            transcript = (record.get("Text") or "").strip()
            audio = record.get("audio") or {}
            if not uid or not transcript or not audio.get("bytes") or uid in seen:
                continue
            seen.add(uid)
            dialect = record.get("Dialect")
            if dialect == "Southwestern":
                southwestern.append((uid, audio["bytes"], transcript))
            elif dialect == "Mandarin":
                mandarin.append((uid, audio["bytes"], transcript))

    logger.info("found %d southwestern, %d mandarin", len(southwestern), len(mandarin))
    stats = {}
    stats["kespeech_sw"] = _dump(output, "kespeech_sw", southwestern, sw_limit)
    stats["kespeech_mandarin"] = _dump(output, "kespeech_mandarin", mandarin, man_limit)
    return stats


def _dump(
    output: Path, name: str, items: list[tuple[str, bytes, str]], limit: int
) -> dict:
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for uid, payload, transcript in items[:limit]:
        path = directory / f"{uid}.wav"
        if not path.exists():
            path.write_bytes(payload)
        rows.append({"wav": str(path), "text": transcript})
    (output / f"{name}.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("wrote %s: %d utts", output / f"{name}.json", len(rows))
    return {"n": len(rows), "json": str(output / f"{name}.json")}


def _parquet_files(source: Path) -> list[Path]:
    shards = sorted(source.glob("test-*-of-*.parquet"))
    if shards:
        return shards
    return sorted(source.glob("test-*.parquet"))
