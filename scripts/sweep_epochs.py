#!/usr/bin/env python3
"""Score every LLM checkpoint on the accent objective instead of on CV loss."""

from __future__ import annotations


from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--spk", default="sichuan_female")
    parser.add_argument("--n-sent", type=int, default=20)
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.evaluation.tts import sweep_checkpoints

    report = sweep_checkpoints(
        settings,
        model_dir=args.model_dir,
        ckpt_dir=args.ckpt_dir,
        out=args.out,
        spk=args.spk,
        n_sent=args.n_sent,
    )
    for tag, values in report.items():
        print(f"{tag:<10} cer_base {values['cer_base']:.4f}  cer_ft {values['cer_ft']:.4f}  rate {values['rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
