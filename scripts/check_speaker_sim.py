#!/usr/bin/env python3
"""Measure whether generated audio carries the requested speaker identity."""

from __future__ import annotations

import argparse

from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--sft-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.evaluation.speaker import check_speaker_similarity

    report = check_speaker_similarity(
        settings, out=args.out, eval_dir=args.eval_dir, sft_dir=args.sft_dir
    )
    for name, values in report.items():
        print(f"{name}: {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
