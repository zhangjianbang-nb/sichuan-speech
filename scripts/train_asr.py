#!/usr/bin/env python3
"""Fine-tune SenseVoiceSmall on the Sichuan manifests.

``--profile round2`` (default) is the recipe that produced the delivered checkpoint:
4 epochs, lr 4e-6, batch 3, 10 s clips, neutral emotion token. ``--profile round1``
reproduces the failed recipe (8 epochs, lr 2e-5, blank emotion token) for the record.
"""

from __future__ import annotations


from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--profile", default="round2", choices=("round1", "round2"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--max-sec", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--smoke", action="store_true", help="few steps, tiny batch, for wiring checks")
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.training.asr import train_asr

    summary = train_asr(
        settings,
        profile=args.profile,
        epochs=args.epochs,
        lr=args.lr,
        batch=args.batch,
        max_sec=args.max_sec,
        out=args.out,
        gpu=args.gpu,
        log_every=args.log_every,
        smoke=args.smoke,
        manifest=args.manifest,
        seed=args.seed,
    )
    for key, value in summary.items():
        print(f"{key:>24}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
