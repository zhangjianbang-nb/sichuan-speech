"""Data preparation helpers that do not need the corpora: annotation cleaning, kaldi I/O."""

from __future__ import annotations

import json
from pathlib import Path

from sichuan_speech.data import clean_annotation, read_manifest, write_kaldi_dir, write_manifest


def test_clean_annotation_keeps_words_and_drops_markers():
    assert clean_annotation("差点【<筐瓢>】") == "差点筐瓢"
    assert clean_annotation("【二天】，我还想多体验") == "二天，我还想多体验"
    assert clean_annotation("没有标记的句子") == "没有标记的句子"
    assert clean_annotation("  【拽实】的队友  ") == "拽实的队友"


def test_write_kaldi_dir_layout(tmp_path: Path):
    rows = [
        ["u1", "/a/u1.wav", "你好", "spk1", 1.5],
        ["u2", "/a/u2.wav", "再见", "spk1", 2.0],
        ["u3", "/a/u3.wav", "早上好", "spk2", 1.0],
    ]
    stats = write_kaldi_dir(tmp_path / "train", rows)
    assert stats["utts"] == 3
    assert stats["speakers"] == 2
    assert stats["hours"] == 0.0  # rounded to two decimals

    directory = tmp_path / "train"
    assert (directory / "wav.scp").read_text(encoding="utf-8").splitlines()[0] == "u1 /a/u1.wav"
    assert (directory / "text").read_text(encoding="utf-8").splitlines()[1] == "u2 再见"
    spk2utt = (directory / "spk2utt").read_text(encoding="utf-8")
    assert "spk1 u1 u2" in spk2utt
    assert "spk2 u3" in spk2utt


def test_utt_ids_with_spaces_are_sanitised(tmp_path: Path):
    write_kaldi_dir(tmp_path / "d", [["u 1", "/a.wav", "你好", "spk", 1.0]])
    assert (tmp_path / "d" / "wav.scp").read_text(encoding="utf-8").startswith("u_1 ")


def test_manifest_round_trip(tmp_path: Path):
    rows = [["u1", "/a.wav", "你好", "spk", 1.0]]
    path = write_manifest(tmp_path / "rows.json", rows)
    assert json.loads(path.read_text(encoding="utf-8")) == rows
    assert read_manifest(path) == rows
