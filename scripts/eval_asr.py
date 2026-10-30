#!/usr/bin/env python3
"""Score the stock recogniser and every fine-tuned checkpoint on the hold-out sets.

Both sides of the comparison go through the same CTC decode path and the same
``use_itn`` setting; mixing them is what produced the fake improvement in round 1.
"""

from __future__ import annotations


from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--ckpt", action="append", default=[], help="checkpoint path; repeatable")
    parser.add_argument("--out", default=None, help="output directory (default: <eval_root>)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-itn", action="store_true", help="enable ITN on both paths (not recommended)")
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.evaluation.asr import evaluate_asr

    ckpts = args.ckpt or [str(settings.asr_ckpt)]
    report = evaluate_asr(settings, ckpts=ckpts, out=args.out, use_itn=args.use_itn, device=args.device)
    for name, values in report.items():
        if isinstance(values, dict):
            print(f"{name}: {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
