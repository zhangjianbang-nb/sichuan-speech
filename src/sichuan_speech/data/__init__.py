"""Data preparation: corpus extraction, manifests and Kaldi-style directories."""

from __future__ import annotations

from .kaldi import read_manifest, write_kaldi_dir, write_manifest
from .kespeech import extract_eval_sets
from .manifest import build_manifests
from .sichuan import build_sichuan_manifest, clean_annotation, label_sichuanhua

__all__ = [
    "build_manifests",
    "build_sichuan_manifest",
    "clean_annotation",
    "extract_eval_sets",
    "label_sichuanhua",
    "read_manifest",
    "write_kaldi_dir",
    "write_manifest",
]
