"""SenseVoiceSmall full-parameter fine-tuning on Sichuan data.

Two profiles reproduce the two rounds that were actually run:

``round2`` (default, the recipe behind the reported numbers)
    lr 4e-6, 4 epochs, batch 3, 10 s cap, clean human transcripts, KeSpeech
    Southwestern + 900 THCHS-30 Mandarin replay, emotion slot = ``<|NEUTRAL|>``.

``round1`` (kept for the post-mortem)
    lr 2e-5, 8 epochs, batch 6, 20 s cap, auto-labelled clips, emotion slot = 0.
    It looked better in-domain but was *worse* than the stock model on the
    cross-speaker held-out set once both sides used ``use_itn=False``.

The label prefix matters. SenseVoice expects
``[<|zh|>, <|Speech|>, <|NEUTRAL|>, <|woitn|>]``; round 1 filled the emotion slot
with 0 (blank) and taught the emotion head to predict blank.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..asr import VOCAB_SPECIAL_START, label_prefix
from ..config import Settings

logger = logging.getLogger(__name__)

PROFILES: dict[str, dict[str, Any]] = {
    "round2": {
        "epochs": 4,
        "lr": 4e-6,
        "batch": 3,
        "max_sec": 10.0,
        "dev_size": 200,
        "neutral": True,
        "manifest": "all_rows_v2.json",
    },
    "round1": {
        "epochs": 8,
        "lr": 2e-5,
        "batch": 6,
        "max_sec": 20.0,
        "dev_size": 160,
        "neutral": False,
        "manifest": "all_rows.json",
    },
}


def _load_wav(path: str) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import torch
        import torchaudio

        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, 16000).numpy()
    return wav


def _make_batch(
    chunk: Sequence[Sequence], frontend: Any, tokenizer: Any, device: str, neutral: bool
):
    import torch

    feats, feat_lens, token_ids, token_lens = [], [], [], []
    prefix = label_prefix(neutral)
    for row in chunk:
        _, path, text = row[0], row[1], row[2]
        wav = _load_wav(str(path))
        tensor = torch.from_numpy(wav).unsqueeze(0)
        feat, feat_len = frontend(tensor, torch.tensor([wav.shape[0]]))
        feats.append(feat[0])
        feat_lens.append(int(feat_len[0]))
        ids = prefix + tokenizer.encode(text)
        token_ids.append(torch.tensor(ids, dtype=torch.long))
        token_lens.append(len(ids))

    max_feat = max(f.shape[0] for f in feats)
    dim = feats[0].shape[1]
    feat_batch = torch.zeros(len(feats), max_feat, dim)
    for index, feat in enumerate(feats):
        feat_batch[index, : feat.shape[0]] = feat

    max_tokens = max(t.shape[0] for t in token_ids)
    token_batch = torch.full((len(token_ids), max_tokens), -1, dtype=torch.long)
    for index, tokens in enumerate(token_ids):
        token_batch[index, : tokens.shape[0]] = tokens

    return (
        feat_batch.to(device),
        torch.tensor(feat_lens, dtype=torch.long).to(device),
        token_batch.to(device),
        torch.tensor(token_lens, dtype=torch.long).to(device),
    )


def _greedy_decode(logits: Any, lens: Any) -> list[list[int]]:
    out = []
    for index in range(logits.shape[0]):
        ids = logits[index, : int(lens[index])].argmax(dim=-1).tolist()
        previous, result = None, []
        for token in ids:
            if token != previous:
                if token != 0 and token < VOCAB_SPECIAL_START:
                    result.append(token)
                previous = token
        out.append(result)
    return out


def _cer_stats(reference: str, hypothesis: str) -> tuple[int, int]:
    from ..text import char_error_stats

    stats = char_error_stats(reference, hypothesis)
    return stats.errors, stats.ref_chars


def train_asr(
    settings: Settings,
    *,
    profile: str = "round2",
    epochs: int | None = None,
    lr: float | None = None,
    batch: int | None = None,
    max_sec: float | None = None,
    out: str | Path | None = None,
    gpu: str | None = None,
    log_every: int = 50,
    smoke: bool = False,
    manifest: str | Path | None = None,
    seed: int = 1986,
) -> dict:
    """Fine-tune SenseVoiceSmall; returns a summary dict."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    config = dict(PROFILES[profile])
    epochs = int(epochs if epochs is not None else config["epochs"])
    lr = float(lr if lr is not None else config["lr"])
    batch = int(batch if batch is not None else config["batch"])
    max_sec = float(max_sec if max_sec is not None else config["max_sec"])
    neutral = bool(config["neutral"])
    out_dir = (
        Path(out)
        if out
        else settings.work_root / ("exp/asr2" if profile == "round2" else "exp/asr")
    )
    manifest_path = (
        Path(manifest) if manifest else settings.data_root / "sft" / str(config["manifest"])
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu or settings.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from funasr import AutoModel
    from funasr.frontends.wav_frontend import WavFrontend

    device = os.environ.get("DEVICE", "cpu" if smoke else "cuda")

    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [row for row in rows if len(row) == 5 and float(row[4]) <= max_sec]
    random.seed(seed)
    random.shuffle(rows)
    dev_size = int(config["dev_size"])
    dev_rows = sorted(rows[:dev_size], key=lambda row: row[0])
    train_rows = rows[dev_size:]
    if smoke:
        train_rows = train_rows[:2]
        dev_rows = dev_rows[:4]

    automodel = AutoModel(model=str(settings.asr_base_dir), device="cpu", disable_update=True)
    model = automodel.model
    tokenizer = automodel.kwargs["tokenizer"]
    frontend = WavFrontend(**automodel.kwargs["frontend_conf"])
    if device == "cuda":
        model = model.cuda()
    model.train()
    logger.info(
        "params %.1fM  train %d  dev %d  profile %s",
        sum(p.numel() for p in model.parameters()) / 1e6,
        len(train_rows),
        len(dev_rows),
        profile,
    )

    def batch_iter(data: list[list], size: int, shuffle: bool = True):
        indices = list(range(len(data)))
        if shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), size):
            yield _make_batch(
                [data[i] for i in indices[start : start + size]],
                frontend,
                tokenizer,
                device,
                neutral,
            )

    @torch.no_grad()
    def evaluate(limit: int | None = None) -> tuple[float, list[tuple[str, str]]]:
        model.eval()
        data = dev_rows[:limit] if limit else dev_rows
        total_errors = total_chars = 0
        predictions: list[tuple[str, str]] = []
        position = 0
        for feat_batch, feat_lens, token_batch, _ in batch_iter(data, 8, shuffle=False):
            encoded, encoded_lens = model.encode(feat_batch, feat_lens, token_batch)
            logits = model.ctc.log_softmax(encoded)
            hypotheses = _greedy_decode(logits, encoded_lens)
            for ids in hypotheses:
                reference = data[position][2]
                position += 1
                hypothesis = tokenizer.decode(ids)
                errors, chars = _cer_stats(reference, hypothesis)
                total_errors += errors
                total_chars += chars
                predictions.append((reference, hypothesis))
        model.train()
        return total_errors / max(1, total_chars), predictions

    if smoke:
        feat_batch, feat_lens, token_batch, token_lens = next(iter(batch_iter(train_rows, 2)))
        loss, stats, weight = model(feat_batch, feat_lens, token_batch, token_lens)
        print("smoke loss", float(loss), "weight", float(weight), stats)
        score, predictions = evaluate()
        for reference, hypothesis in predictions:
            print("REF:", reference)
            print("HYP:", hypothesis)
        print("smoke cer", round(score, 4))
        return {"profile": profile, "smoke": True, "dev_cer": round(score, 4)}

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_rows) / batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=steps_per_epoch * epochs, pct_start=0.05
    )

    history = []
    started = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        batches = 0
        for feat_batch, feat_lens, token_batch, token_lens in batch_iter(train_rows, batch):
            loss, stats, weight = model(feat_batch, feat_lens, token_batch, token_lens)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss)
            batches += 1
            if batches % log_every == 0:
                logger.info(
                    "ep %d step %d/%d loss %.4f lr %.2e elapsed %.0fs",
                    epoch,
                    batches,
                    steps_per_epoch,
                    running_loss / batches,
                    scheduler.get_last_lr()[0],
                    time.time() - started,
                )
        dev_cer, predictions = evaluate()
        logger.info(
            "EPOCH %d avg_loss %.4f DEV_CER %.4f", epoch, running_loss / max(1, batches), dev_cer
        )
        for reference, hypothesis in predictions[:2]:
            logger.info("  REF: %s", reference)
            logger.info("  HYP: %s", hypothesis)
        history.append({"epoch": epoch, "loss": running_loss / max(1, batches), "dev_cer": dev_cer})
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "cer": dev_cer},
            out_dir / f"epoch_{epoch}.pt",
        )

    best = min(history, key=lambda item: item["dev_cer"])
    summary = {
        "profile": profile,
        "epochs": history,
        "best_epoch": best["epoch"],
        "best_dev_cer": round(best["dev_cer"], 4),
        "out": str(out_dir),
        "train_utts": len(train_rows),
        "dev_utts": len(dev_rows),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info("DONE_ASR %s", json.dumps(summary, ensure_ascii=False))
    return summary
