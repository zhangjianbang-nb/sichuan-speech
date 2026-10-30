#!/usr/bin/env python3
"""The s/sh merger test — reproduces a failed metric and writes why it failed."""

from __future__ import annotations


from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--sft-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--per-set", type=int, default=25)
    parser.add_argument("--n-takes", type=int, default=3)
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.evaluation.minshi import evaluate_minshi

    report = evaluate_minshi(settings, sft_dir=args.sft_dir, out=args.out, per_set=args.per_set, n_takes=args.n_takes)
    for name, values in report.items():
        print(f"{name}: {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
