#!/usr/bin/env python3
"""Assemble an inference-ready CosyVoice2 directory from an SFT checkpoint.

Symlinks the untouched flow/vocoder files, writes ``llm.pt``, registers the fine-tuned
speaker and the averaged speaker, and copies ``llm_embedding`` to ``embedding`` because
``frontend_sft`` reads the latter key while ``add_zero_shot_spk`` writes the former.
"""

from __future__ import annotations

from _common import build_parser, settings_from_args


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--ckpt", required=True, help="epoch_N_whole.pt from the SFT run")
    parser.add_argument(
        "--out",
        default=None,
        help="destination model dir (default: <work_root>/models/cosyvoice2-sichuan)",
    )
    parser.add_argument("--prompt-wav", default=None)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--spk-id", default="sichuan_female")
    parser.add_argument(
        "--emb-pt", default=None, help="spk2embedding.pt (default: <data_root>/sft/train)"
    )
    parser.add_argument("--utt2spk", default=None, help="utt2spk (default: <data_root>/sft/train)")
    parser.add_argument("--spk-prefix", default="ks_")
    parser.add_argument("--avg-spk-id", default="sichuan_avg")
    args = parser.parse_args()
    settings = settings_from_args(args)

    from sichuan_speech.training.tts import build_inference_model

    train_dir = settings.data_root / "sft" / "train"
    summary = build_inference_model(
        settings,
        ckpt=args.ckpt,
        out=args.out or str(settings.work_root / "models" / "cosyvoice2-sichuan"),
        prompt_wav=args.prompt_wav or str(settings.prompt_wav),
        prompt_text=args.prompt_text or settings.prompt_text,
        spk_id=args.spk_id,
        emb_pt=args.emb_pt or str(train_dir / "spk2embedding.pt"),
        utt2spk=args.utt2spk or str(train_dir / "utt2spk"),
        spk_prefix=args.spk_prefix,
        avg_spk_id=args.avg_spk_id,
    )
    for key, value in summary.items():
        print(f"{key:>24}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
