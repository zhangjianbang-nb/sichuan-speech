#!/usr/bin/env python3
"""Extract the KeSpeech evaluation splits (400 Southwestern + 300 Mandarin)."""

from __future__ import annotations

import argparse

from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--sw-limit", type=int, default=400)
    parser.add_argument("--man-limit", type=int, default=300)
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.data import extract_eval_sets

    summary = extract_eval_sets(settings, sw_limit=args.sw_limit, man_limit=args.man_limit)
    for key, value in summary.items():
        print(f"{key:>24}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
