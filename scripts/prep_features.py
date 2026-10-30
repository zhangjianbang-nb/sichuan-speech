#!/usr/bin/env python3
"""Run CosyVoice's feature-extraction tools: speaker embedding, speech tokens, parquet.

Three passes per split, ~13 minutes for 9.12 h of audio on one 4090. The tools live in
``<cosyvoice_home>/tools`` (not under ``examples/``) and must run with the CosyVoice
root and ``third_party/Matcha-TTS`` on ``PYTHONPATH``.
"""

from __future__ import annotations


from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--python", default=None, help="interpreter used for the tools (default: current)")
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--num-processes", type=int, default=10)
    parser.add_argument("--utts-per-parquet", type=int, default=1000)
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.training.tts import prepare_features

    summary = prepare_features(
        settings,
        python=args.python,
        splits=tuple(args.splits),
        num_processes=args.num_processes,
        utts_per_parquet=args.utts_per_parquet,
    )
    for key, value in summary.items():
        print(f"{key:>24}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
