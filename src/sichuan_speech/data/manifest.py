"""Build the training manifests.

Round 2 (the recipe that produced the reported numbers) uses only human
transcripts:

* Sichuan: MagicHub Chengdu female (78) + liugztat Southwestern Mandarin (1775)
* KeSpeech Southwestern Mandarin, human text, minus the 400 held-out eval utts
* Mandarin replay: 900 THCHS-30 utterances (18% of the mix) to prevent forgetting

The round-1 manifest additionally included 730 ASR-auto-labelled clips whose
transcripts contained recognition errors; those taught the TTS wrong
text-to-speech alignments and were dropped. Use ``legacy=True`` only to
reproduce round 1.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterable

from ..config import Settings
from .kaldi import write_kaldi_dir, write_manifest

logger = logging.getLogger(__name__)


def build_manifests(
    settings: Settings,
    *,
    legacy: bool = False,
    dev_ratio: float = 0.03,
    mandarin_cap: int = 900,
    seed: int = 1986,
    thchs_parquet: str | Path | None = None,
) -> dict:
    """Build ``data/sft/{train,dev}`` and the JSON manifest.

    Returns a summary dict with utterance counts and hours per source.
    """
    data_root = settings.data_root
    sft_root = data_root / "sft"
    sft_root.mkdir(parents=True, exist_ok=True)
    rows: list[list] = []

    # 1) human-transcribed Sichuan
    sichuan_file = sft_root / "all_sichuan.json"
    if not sichuan_file.exists():
        from .sichuan import build_sichuan_manifest

        logger.info("%s missing; building it from the raw corpora", sichuan_file)
        build_sichuan_manifest(settings, out=sichuan_file)
    for row in json.loads(sichuan_file.read_text(encoding="utf-8")):
        if len(row) == 5:
            rows.append(list(row))
    n_sichuan = len(rows)
    logger.info("clean Sichuan: %d utts", n_sichuan)

    # 2a) round-1 extra: ASR-auto-labelled clips (noisy, kept only for legacy)
    if legacy:
        labels_file = data_root / "sichuanhua_asr.json"
        if labels_file.exists():
            labels = json.loads(labels_file.read_text(encoding="utf-8"))
            raw_dir = data_root / "raw" / "sichuanhua"
            before = len(rows)
            for basename, transcript in labels.items():
                path = raw_dir / basename
                seconds = _duration(path)
                if 1.0 <= seconds <= 25 and len(transcript) >= 2:
                    rows.append(
                        [basename[:-4], str(path), transcript, f"utt_{basename[:-4]}", seconds]
                    )
            logger.info("auto-labelled Sichuan added: %d", len(rows) - before)

    # 2b) round-2 extra: KeSpeech Southwestern Mandarin with human transcripts
    if not legacy:
        added = _add_kespeech(settings, rows)
        logger.info("KeSpeech Southwestern added: %d", added)

    # 3) Mandarin replay (THCHS-30)
    added = _add_thchs(settings, rows, cap=mandarin_cap, seed=7, parquet=thchs_parquet)
    logger.info("Mandarin replay: %d", added)

    random.seed(seed)
    random.shuffle(rows)
    n_dev = max(20, int(len(rows) * dev_ratio))
    dev, train = rows[:n_dev], rows[n_dev:]

    stats = {}
    for name, subset in (("train", train), ("dev", dev)):
        stats[name] = write_kaldi_dir(sft_root / name, subset)

    manifest_name = "all_rows.json" if legacy else "all_rows_v2.json"
    write_manifest(sft_root / manifest_name, rows)

    total_hours = sum(float(row[4]) for row in rows) / 3600.0
    summary = {
        "sichuan": n_sichuan,
        "total_utts": len(rows),
        "total_hours": round(total_hours, 2),
        "train": stats["train"],
        "dev": stats["dev"],
        "manifest": str(sft_root / manifest_name),
        "legacy": legacy,
    }
    logger.info(
        "TOTAL %d utts / %.2f h (train %d, dev %d)",
        summary["total_utts"],
        summary["total_hours"],
        stats["train"]["utts"],
        stats["dev"]["utts"],
    )
    return summary


def _add_kespeech(settings: Settings, rows: list[list]) -> int:
    """Extract KeSpeech Southwestern audio, excluding the held-out eval ids."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required for KeSpeech extraction") from exc

    eval_file = settings.data_root / "eval" / "kespeech_sw.json"
    eval_ids: set[str] = set()
    if eval_file.exists():
        eval_ids = {
            Path(row["wav"]).stem
            for row in json.loads(eval_file.read_text(encoding="utf-8"))
        }

    source_dir = settings.data_root / "raw" / "kespeech"
    output_dir = settings.data_root / "raw" / "kespeech_sw_train"
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = _kespeech_parquet_files(source_dir)
    if not parquet_files:
        raise FileNotFoundError(f"no KeSpeech parquet files under {source_dir}")

    seen: set[str] = set()
    added = 0
    for parquet in parquet_files:
        for record in pq.read_table(parquet).to_pylist():
            if record.get("Dialect") != "Southwestern":
                continue
            transcript = (record.get("Text") or "").strip()
            uid = record.get("ID") or ""
            if not transcript or not uid or uid in eval_ids or uid in seen:
                continue
            seen.add(uid)
            path = output_dir / f"{uid}.wav"
            if not path.exists():
                audio = record.get("audio") or {}
                if audio.get("bytes"):
                    path.write_bytes(audio["bytes"])
            seconds = _duration(path)
            if 1.0 <= seconds <= 25 and len(transcript) >= 2:
                rows.append([uid, str(path), transcript, "ks_" + uid.split("_")[0], seconds])
                added += 1
    return added


def _add_thchs(
    settings: Settings,
    rows: list[list],
    *,
    cap: int,
    seed: int,
    parquet: str | Path | None = None,
) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required for THCHS-30 extraction") from exc

    source = Path(parquet) if parquet else settings.data_root / "raw" / "mandarin" / "thchs_train0.parquet"
    if not source.exists():
        logger.warning("THCHS-30 parquet not found at %s; skipping Mandarin replay", source)
        return 0
    wav_dir = settings.data_root / "raw" / "mandarin" / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[list] = []
    for record in pq.read_table(source).to_pylist():
        filename = Path(str(record.get("audio", {}).get("path") or "")).name
        if not filename:
            continue
        path = wav_dir / filename
        if not path.exists():
            payload = record.get("audio", {}).get("bytes")
            if not payload:
                continue
            path.write_bytes(payload)
        seconds = _duration(path)
        transcript = str(record.get("sentence") or "")
        if 1.0 <= seconds <= 25 and len(transcript) >= 2:
            candidates.append(
                [filename[:-4], str(path), transcript, "thchs_" + filename.split("_")[0], seconds]
            )

    random.seed(seed)
    random.shuffle(candidates)
    chosen = candidates[:cap]
    rows.extend(chosen)
    return len(chosen)


def _kespeech_parquet_files(source_dir: Path) -> list[Path]:
    """Prefer the complete 8-shard split; fall back to whatever exists."""
    shards = sorted(source_dir.glob("test-*-of-*.parquet"))
    if shards:
        return shards
    return sorted(source_dir.glob("test-*.parquet"))


def _duration(path: str | Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001 - unreadable file counts as unusable
        return 0.0


def iter_rows(path: str | Path) -> Iterable[list]:
    """Iterate manifest rows from JSON."""
    for row in json.loads(Path(path).read_text(encoding="utf-8")):
        yield list(row)
