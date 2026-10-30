"""Build ``data/sft/all_sichuan.json`` from the two human-transcribed Sichuan corpora.

Sources (both downloaded with ``hf-mirror`` and unpacked under ``<data_root>/raw``):

* ``magic_tts/txt.txt`` — MagicHub Chengdu female, one ``<file>.wav<TAB>text`` per line.
* ``xiNan_text.txt`` — liugztat Southwestern Mandarin, ``file<TAB>annotated text<TAB>keywords``.
  The annotation wraps dialect words in 【】 and sometimes adds a pronunciation hint in
  <>; only the words are kept, the markers are dropped, e.g.
  ``差点【<筐瓢>】`` becomes ``差点筐瓢``.

The resulting JSON is the input to :func:`sichuan_speech.data.manifest.build_manifests`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import Settings

logger = logging.getLogger(__name__)

BRACKETS = str.maketrans(dict.fromkeys("【】<>「」『』"))
MAGIC_SPEAKER = "spk_chengdu_f"


def clean_annotation(text: str) -> str:
    """Drop annotation markers while keeping the annotated words."""
    return text.translate(BRACKETS).strip()


def _duration(path: Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        return 0.0


def _read_tsv(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        rows.append((parts[0].strip(), parts[1].strip()))
    return rows


def build_sichuan_manifest(
    settings: Settings,
    *,
    out: str | Path | None = None,
    min_seconds: float = 1.0,
    max_seconds: float = 25.0,
) -> dict:
    """Write ``all_sichuan.json``; returns counts per source."""
    raw = settings.data_root / "raw"
    rows: list[list] = []
    counts: dict[str, int] = {}

    magic_dir = raw / "magic_tts"
    magic_index = magic_dir / "txt.txt"
    if magic_index.exists():
        before = len(rows)
        for filename, text in _read_tsv(magic_index):
            path = magic_dir / filename
            if not path.exists():
                continue
            seconds = _duration(path)
            if not min_seconds <= seconds <= max_seconds or len(text) < 2:
                continue
            rows.append([path.stem, str(path), text, MAGIC_SPEAKER, seconds])
        counts["magic_tts"] = len(rows) - before
    else:
        counts["magic_tts"] = 0
        logger.warning("missing %s; skipping MagicHub Chengdu corpus", magic_index)

    xinan_index = raw / "xiNan_text.txt"
    xinan_dir = raw / "xiNan"
    if xinan_index.exists():
        before = len(rows)
        for filename, annotated in _read_tsv(xinan_index):
            text = clean_annotation(annotated)
            path = xinan_dir / filename
            if not path.exists():
                continue
            seconds = _duration(path)
            if not min_seconds <= seconds <= max_seconds or len(text) < 2:
                continue
            rows.append([path.stem, str(path), text, "utt_" + path.stem, seconds])
        counts["xinan"] = len(rows) - before
    else:
        counts["xinan"] = 0
        logger.warning("missing %s; skipping liugztat Southwestern corpus", xinan_index)

    target = Path(out) if out else settings.data_root / "sft" / "all_sichuan.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    summary = {
        "total": len(rows),
        "hours": round(sum(float(row[4]) for row in rows) / 3600.0, 2),
        "manifest": str(target),
        **counts,
    }
    logger.info(
        "wrote %s: %d utts / %.2f h (%s)", target, summary["total"], summary["hours"], counts
    )
    return summary


def label_sichuanhua(
    settings: Settings,
    *,
    out: str | Path | None = None,
    limit: int | None = None,
) -> dict:
    """Round-1 only: auto-label ``aotemande/sichuanhua`` with the stock recogniser.

    Kept for reproducibility, not for training. The labels contain recognition errors
    (医万 for 医院, 真溪 for 珍溪); round 2 dropped the whole corpus for that reason.
    """
    from ..asr import load_recognizer

    source = settings.data_root / "raw" / "sichuanhua"
    files = sorted(source.glob("*.wav"))[: limit or None]
    if not files:
        raise FileNotFoundError(f"no wav files under {source}")
    recognizer = load_recognizer(settings, finetuned=False)
    labels: dict[str, str] = {}
    for index, path in enumerate(files, start=1):
        labels[path.name] = recognizer.transcribe(str(path))
        if index % 25 == 0:
            logger.info("labelled %d/%d", index, len(files))
    target = Path(out) if out else settings.data_root / "sichuanhua_asr.json"
    target.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("wrote %s (%d clips)", target, len(labels))
    return {"clips": len(labels), "manifest": str(target)}
