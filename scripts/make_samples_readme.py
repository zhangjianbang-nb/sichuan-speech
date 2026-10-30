#!/usr/bin/env python3
"""Regenerate ``samples/transcripts.json`` and ``samples/README.md`` from the demo wavs."""

from __future__ import annotations


from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--dir", default=None, help="directory holding the demo wavs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.evaluation.samples import build_samples_readme

    rows = build_samples_readme(settings, directory=args.dir, out=args.out)
    print(f"{len(rows)} samples scored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
