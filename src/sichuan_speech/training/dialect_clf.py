"""Dialect classifiers.

Two classifiers were trained. Both are kept because the failure of the first one
is part of the evidence trail (see ``docs/evaluation.md`` §4.1):

``train_kespeech_classifier``
    KeSpeech Southwestern vs Mandarin, channel-matched, speaker-disjoint
    ``GroupKFold``. Speaker-disjoint accuracy was only **0.567** (chance 0.5):
    without grouping it reaches 0.9+, which is speaker/channel memorisation, not
    dialect. Kept as a documented negative result.

``train_dialect_classifier``
    Log-mel CNN over MagicHub/xiNan Sichuan vs THCHS-30 Mandarin. Same caveat:
    cross-corpus accuracy is domain-shifted, so it is only meaningful as a
    *relative* score between two synthesised sets sharing one vocoder.
"""

from __future__ import annotations

import glob
import json
import logging
import random
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import Settings

logger = logging.getLogger(__name__)

N_MELS = 64
MAX_FRAMES = 400
SR = 16000


@lru_cache(maxsize=1)
def _net_class():
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 8)),
            )
            self.f = nn.Linear(64 * 4 * 8, 2)

        def forward(self, x):  # noqa: ANN001, ANN201 - torch module signature
            return self.f(self.c(x).flatten(1))

    return Net


def _librosa_mel(path: str | Path, *, max_frames: int = MAX_FRAMES) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(str(path), sr=SR)
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SR, n_fft=1024, hop_length=160, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel + 40.0) / 40.0
    if mel.shape[1] > max_frames:
        mel = mel[:, :max_frames]
    else:
        mel = np.pad(mel, ((0, 0), (0, max_frames - mel.shape[1])), constant_values=-1.0)
    return mel.astype(np.float32)


def train_kespeech_classifier(
    settings: Settings,
    *,
    out: str | Path | None = None,
    folds: int = 5,
    epochs: int = 12,
    final_epochs: int = 20,
    seed: int = 1986,
) -> dict:
    """Speaker-disjoint CV then a final fit; returns the accuracy summary."""
    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupKFold

    torch.manual_seed(seed)
    eval_root = settings.data_root / "eval"
    rows: list[tuple[str, int]] = [
        (row["wav"], 1) for row in json.loads((eval_root / "kespeech_sw.json").read_text(encoding="utf-8"))
    ] + [
        (row["wav"], 0)
        for row in json.loads((eval_root / "kespeech_mandarin.json").read_text(encoding="utf-8"))
    ]
    features = np.stack([_librosa_mel(path) for path, _ in rows])[:, None]
    labels = np.array([label for _, label in rows])
    groups = np.array([Path(path).stem.split("_")[0] for path, _ in rows])
    logger.info(
        "samples %d, speakers %d, southwestern speakers %d",
        len(labels),
        len(set(groups)),
        len({s for s, label in zip(groups, labels) if label == 1}),
    )

    Net = _net_class()
    accuracies = []
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(features, labels, groups):
        net = Net()
        optimizer = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        train_x = torch.from_numpy(features[train_idx])
        train_y = torch.from_numpy(labels[train_idx])
        for _ in range(epochs):
            net.train()
            permutation = torch.randperm(len(train_x))
            for start in range(0, len(train_x), 32):
                batch = permutation[start : start + 32]
                optimizer.zero_grad()
                loss = loss_fn(net(train_x[batch]), train_y[batch])
                loss.backward()
                optimizer.step()
        net.eval()
        with torch.no_grad():
            predicted = net(torch.from_numpy(features[test_idx])).argmax(-1).numpy()
        accuracy = float((predicted == labels[test_idx]).mean())
        accuracies.append(accuracy)
        logger.info("fold acc %.3f (n=%d)", accuracy, len(test_idx))

    net = Net()
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    train_x = torch.from_numpy(features)
    train_y = torch.from_numpy(labels)
    for _ in range(final_epochs):
        net.train()
        permutation = torch.randperm(len(train_x))
        for start in range(0, len(train_x), 32):
            batch = permutation[start : start + 32]
            optimizer.zero_grad()
            loss = loss_fn(net(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()

    target = Path(out) if out else settings.work_root / "clf_sw.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), target)
    summary = {
        "cv_mean": round(float(np.mean(accuracies)), 3),
        "cv_std": round(float(np.std(accuracies)), 3),
        "folds": accuracies,
        "saved": str(target),
        "note": "speaker-disjoint CV; near chance means the classifier learns channel, not dialect",
    }
    logger.info("speaker-disjoint CV acc %.3f +- %.3f", summary["cv_mean"], summary["cv_std"])
    return summary


def _torch_mel(wav: np.ndarray):
    import torch
    import torchaudio

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=400, hop_length=160, n_mels=N_MELS, f_min=0, f_max=8000
    )
    tensor = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    return torch.log(transform(tensor).clamp(min=1e-6))


def _load_segment(path: str | Path, seconds: float = 3.0, *, random_start: bool = False) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio

    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    length = int(seconds * SR)
    if len(wav) < length:
        wav = np.pad(wav, (0, length - len(wav)))
    start = random.randint(0, max(0, len(wav) - length)) if random_start else 0
    return wav[start : start + length]


def _collect_items(settings: Settings) -> list[tuple[str, int]]:
    sichuan: list[tuple[str, int]] = []
    manifest = settings.data_root / "sft" / "all_sichuan.json"
    if manifest.exists():
        for row in json.loads(manifest.read_text(encoding="utf-8")):
            if len(row) == 5:
                sichuan.append((row[1], 1))
    labels_file = settings.data_root / "sichuanhua_asr.json"
    if labels_file.exists():
        for basename in json.loads(labels_file.read_text(encoding="utf-8")):
            sichuan.append((str(settings.data_root / "raw" / "sichuanhua" / basename), 1))
    mandarin = [
        (path, 0) for path in glob.glob(str(settings.data_root / "raw" / "mandarin" / "wav" / "*.wav"))
    ]
    random.shuffle(sichuan)
    random.shuffle(mandarin)
    limit = min(len(sichuan), len(mandarin), 1600)
    items = sichuan[:limit] + mandarin[:limit]
    random.shuffle(items)
    return items


def train_dialect_classifier(
    settings: Settings,
    *,
    out_dir: str | Path | None = None,
    epochs: int = 6,
    gpu: str | None = None,
    seed: int = 1986,
    batch_size: int = 16,
) -> dict:
    """Train the log-mel CNN used by the accent evaluation."""
    import torch
    import torch.nn as nn

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu or settings.gpu)
    torch.manual_seed(seed)
    random.seed(seed)

    items = _collect_items(settings)
    n_val = int(len(items) * 0.1)
    val, train = items[:n_val], items[n_val:]
    logger.info("train %d, val %d", len(train), len(val))

    def collate(batch: Sequence[tuple[str, int]]):
        features = [_torch_mel(_load_segment(path, random_start=True)) for path, _ in batch]
        labels = torch.tensor([label for _, label in batch])
        max_frames = max(feature.shape[1] for feature in features)
        padded = torch.zeros(len(features), N_MELS, max_frames)
        for index, feature in enumerate(features):
            padded[index, :, : feature.shape[1]] = feature
        return padded, labels

    Net = _net_class()
    model = Net().cuda() if torch.cuda.is_available() else Net()
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        random.shuffle(train)
        total = correct = 0
        running = 0.0
        for start in range(0, len(train), batch_size):
            x, y = collate(train[start : start + batch_size])
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for start in range(0, len(val), batch_size):
                x, y = collate(val[start : start + batch_size])
                predicted = model(x.to(device)).argmax(1).cpu()
                val_correct += int((predicted == y).sum())
                val_total += len(y)
        logger.info(
            "ep %d loss %.4f train_acc %.3f val_acc %.3f",
            epoch,
            running / max(1, len(train) // batch_size + 1),
            correct / max(1, total),
            val_correct / max(1, val_total),
        )

    target = Path(out_dir) if out_dir else settings.work_root / "exp" / "dialect_clf"
    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target / "clf.pt")
    summary = {
        "train": len(train),
        "val": len(val),
        "val_acc": round(val_correct / max(1, val_total), 3),
        "saved": str(target / "clf.pt"),
    }
    return summary


def evaluate_dialect_classifier(
    settings: Settings,
    *,
    clf_path: str | Path | None = None,
    sets: Sequence[str] = ("kespeech_sw", "kespeech_mandarin"),
    device: str = "cpu",
) -> dict:
    """Cross-corpus sanity check for the log-mel CNN."""
    import torch

    target = Path(clf_path) if clf_path else settings.work_root / "exp" / "dialect_clf" / "clf.pt"
    if not target.exists():
        raise FileNotFoundError(f"classifier not found: {target}")
    Net = _net_class()
    net = Net()
    net.load_state_dict(torch.load(str(target), map_location=device, weights_only=True))
    net.to(device).eval()

    results = {}
    for name in sets:
        rows = json.loads((settings.data_root / "eval" / f"{name}.json").read_text(encoding="utf-8"))
        probabilities = []
        with torch.no_grad():
            for row in rows:
                wav = _load_segment(row["wav"], seconds=3.0)
                mel = _torch_mel(wav).unsqueeze(0).unsqueeze(0).to(device)
                probabilities.append(float(torch.softmax(net(mel), 1)[0, 1]))
        mean = float(np.mean(probabilities))
        results[name] = {
            "n": len(probabilities),
            "mean_p_sichuan": round(mean, 3),
            "frac_gt_0.5": round(float(np.mean([p > 0.5 for p in probabilities])), 3),
        }
        logger.info("%s n=%d mean_p=%.3f frac>0.5=%.3f", name, len(probabilities), mean,
                    results[name]["frac_gt_0.5"])
    return results
