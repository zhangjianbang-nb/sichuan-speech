#!/usr/bin/env python3
"""Build ``data/sft/{train,dev}`` Kaldi directories and the JSON manifests.

Round 2 (default) mixes human-transcribed Sichuan, KeSpeech Southwestern, and 900
THCHS-30 Mandarin replays; ``--legacy`` reproduces round 1 (auto-labelled Sichuan
only), which is kept so the round-1 numbers stay reproducible.
"""

from __future__ import annotations

from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--legacy", action="store_true", help="round-1 recipe")
    parser.add_argument("--dev-ratio", type=float, default=0.03)
    parser.add_argument("--mandarin-cap", type=int, default=900)
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--thchs-parquet", default=None)
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.data import build_manifests

    summary = build_manifests(
        settings,
        legacy=args.legacy,
        dev_ratio=args.dev_ratio,
        mandarin_cap=args.mandarin_cap,
        seed=args.seed,
        thchs_parquet=args.thchs_parquet,
    )
    for key, value in summary.items():
        print(f"{key:>24}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
