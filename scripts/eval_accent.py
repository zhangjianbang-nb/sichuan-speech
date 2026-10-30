#!/usr/bin/env python3
"""Accent calibration against real Sichuan and real Mandarin anchors."""

from __future__ import annotations


from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--sft-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--clf", default=None)
    parser.add_argument("--n-sent", type=int, default=10)
    parser.add_argument("--n-anchor", type=int, default=40)
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.evaluation.accent import evaluate_accent

    report = evaluate_accent(
        settings,
        sft_dir=args.sft_dir,
        out=args.out,
        clf=args.clf,
        n_sent=args.n_sent,
        n_anchor=args.n_anchor,
    )
    for name, values in report.items():
        print(f"{name:<24} rate {values.get('rate_syl_s')}  f0_med {values.get('f0_med_semi')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
