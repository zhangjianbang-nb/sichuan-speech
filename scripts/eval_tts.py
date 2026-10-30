#!/usr/bin/env python3
"""Full TTS evaluation: anchors, four synthesised sets, Mandarin regression guard."""

from __future__ import annotations

from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--sft-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-anchor", type=int, default=40)
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.evaluation.tts import evaluate_tts

    report = evaluate_tts(settings, sft_dir=args.sft_dir, out=args.out, n_anchor=args.n_anchor)
    for name, values in report.items():
        if isinstance(values, dict):
            print(f"{name:<24} CER {values.get('mean_cer', '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
