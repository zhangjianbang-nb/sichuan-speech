#!/usr/bin/env python3
"""Train the log-mel CNN dialect classifier used by the accent evaluation.

The classifier is a documented failure as an evaluation metric: speaker-disjoint
cross-validation accuracy is 0.567 (chance 0.5). ``--kespeech-cv`` runs that
speaker-disjoint protocol; the default trains the model the accent script loads.
"""

from __future__ import annotations

from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument(
        "--kespeech-cv", action="store_true", help="run the speaker-disjoint CV protocol"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1986)
    args = parser.parse_args()
    settings = settings_from_args(args)

    if args.kespeech_cv:
        from sichuan_speech.training.dialect_clf import train_kespeech_classifier

        summary = train_kespeech_classifier(
            settings, out=args.out, folds=args.folds, seed=args.seed
        )
    else:
        from sichuan_speech.training.dialect_clf import train_dialect_classifier

        summary = train_dialect_classifier(
            settings,
            out_dir=args.out,
            epochs=args.epochs,
            gpu=args.gpu,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    for key, value in summary.items():
        print(f"{key:>24}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
