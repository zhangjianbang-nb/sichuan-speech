#!/usr/bin/env python3
"""Two recognisers score the same synthesised audio (auxiliary observation only)."""

from __future__ import annotations

import argparse

from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--eval-dir", default=None, help="directory holding the synthesised subsets")
    parser.add_argument("--ft-ckpt", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.evaluation.tts import cross_asr

    report = cross_asr(settings, out=args.out, ft_ckpt=args.ft_ckpt, eval_dir=args.eval_dir)
    for name, values in report.items():
        print(f"{name:<24} base {values['cer_base']:.4f}  ft {values['cer_ft']:.4f}  delta {values['delta']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
