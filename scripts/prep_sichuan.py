#!/usr/bin/env python3
"""Build ``data/sft/all_sichuan.json`` from the human-transcribed Sichuan corpora.

``--label-sichuanhua`` additionally auto-labels aotemande/sichuanhua with the stock
recogniser; that path exists only to reproduce round 1, whose labels were too noisy to
train on (see docs/training-notes.md).
"""

from __future__ import annotations

from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--out", default=None)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=25.0)
    parser.add_argument("--label-sichuanhua", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.data import build_sichuan_manifest, label_sichuanhua

    summary = build_sichuan_manifest(
        settings, out=args.out, min_seconds=args.min_seconds, max_seconds=args.max_seconds
    )
    for key, value in summary.items():
        print(f"{key:>24}: {value}")

    if args.label_sichuanhua:
        labelled = label_sichuanhua(settings, limit=args.limit)
        for key, value in labelled.items():
            print(f"{key:>24}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
