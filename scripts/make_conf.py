#!/usr/bin/env python3
"""Derive the CosyVoice2 SFT config from the official template.

Only three edits are applied: ``use_spk_embedding: True``, a constant low learning
rate (3e-6; 1e-5 overfits after one epoch) and denser logging.
"""

from __future__ import annotations

from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument(
        "--out", default=None, help="destination yaml (default: <work_root>/conf/sft.yaml)"
    )
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--max-epoch", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.training.tts import derive_sft_config

    destination = args.out or str(settings.work_root / "conf" / "sft.yaml")
    path = derive_sft_config(
        settings.cosyvoice_home,
        destination,
        lr=args.lr,
        max_epoch=args.max_epoch,
        log_interval=args.log_interval,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
