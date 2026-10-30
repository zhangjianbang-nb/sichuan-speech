"""Shared plumbing for the ``scripts/`` entry points.

These scripts are thin argparse wrappers around :mod:`sichuan_speech`; they add the
repo ``src`` directory to ``sys.path`` so they run from a checkout without installing,
and they take every path from the environment-driven :class:`~sichuan_speech.config.Settings`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sichuan_speech.config import Settings  # noqa: E402
from sichuan_speech.logging_utils import configure_logging  # noqa: E402


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value (default: settings)")
    parser.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    configure_logging(args.log_level)
    settings = Settings.from_env()
    if args.gpu:
        settings = settings.with_overrides(gpu=args.gpu)
    settings.configure_gpu()
    return settings
