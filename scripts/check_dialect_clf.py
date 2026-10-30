#!/usr/bin/env python3
"""Cross-corpus sanity check for the dialect classifier (documents why it is unusable)."""

from __future__ import annotations


from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--clf", default=None)
    parser.add_argument("--sets", nargs="+", default=["kespeech_sw", "kespeech_mandarin"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.training.dialect_clf import evaluate_dialect_classifier

    summary = evaluate_dialect_classifier(
        settings, clf_path=args.clf, sets=tuple(args.sets), device=args.device
    )
    for name, values in summary.items():
        print(f"{name}: {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
