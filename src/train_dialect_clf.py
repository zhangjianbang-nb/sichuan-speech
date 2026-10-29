"""Sichuan-vs-Mandarin dialect classifier (log-mel CNN), used to score TTS output."""
import os, json, glob, random, argparse
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/chuan/exp/dialect_clf"))
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--gpu", default="1")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(1986)
random.seed(1986)

from dialect_clf_lib import Net, mel, SR, N_MELS, HOP, WIN


def load(path, sec=3.0):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    n = int(sec * SR)
    if len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    start = random.randint(0, max(0, len(wav) - n)) if len(wav) > n else 0
    return mel(wav[start:start + n])


def build_items():
    sic = []
    for name in ["all_sichuan.json"]:
        for r in json.load(open(os.path.expanduser("~/chuan/data/sft/" + name))):
            if len(r) == 5:
                sic.append((r[1], 1))
    lab = json.load(open(os.path.expanduser("~/chuan/data/sichuanhua_asr.json")))
    for b in lab:
        sic.append((os.path.expanduser("~/chuan/data/raw/sichuanhua/" + b), 1))
    man = []
    for p in glob.glob(os.path.expanduser("~/chuan/data/raw/mandarin/wav/*.wav")):
        man.append((p, 0))
    random.shuffle(sic)
    random.shuffle(man)
    n = min(len(sic), len(man), 1600)
    items = sic[:n] + man[:n]
    random.shuffle(items)
    return items


items = build_items()
n_val = int(len(items) * 0.1)
val, tr = items[:n_val], items[n_val:]
print("train", len(tr), "val", len(val), flush=True)


import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def collate(batch):
    xs = [load(p) for p, _ in batch]
    ys = torch.tensor([y for _, y in batch])
    maxf = max(x.shape[1] for x in xs)
    out = torch.zeros(len(xs), N_MELS, maxf)
    for i, x in enumerate(xs):
        out[i, :, :x.shape[1]] = x
    return out, ys


model = Net().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss()
bs = 16
for ep in range(args.epochs):
    model.train()
    random.shuffle(tr)
    tot = cor = 0
    run = 0.0
    for i in range(0, len(tr), bs):
        x, y = collate(tr[i:i + bs])
        x, y = x.cuda(), y.cuda()
        logit = model(x)
        loss = lossf(logit, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        run += float(loss)
        cor += int((logit.argmax(1) == y).sum())
        tot += len(y)
    model.eval()
    vtot = vcor = 0
    with torch.no_grad():
        for i in range(0, len(val), bs):
            x, y = collate(val[i:i + bs])
            pred = model(x.cuda()).argmax(1).cpu()
            vcor += int((pred == y).sum())
            vtot += len(y)
    print("ep %d loss %.4f train_acc %.3f val_acc %.3f" % (
        ep, run / max(1, len(tr) // bs + 1), cor / tot, vcor / vtot), flush=True)
torch.save(model.state_dict(), os.path.join(args.out, "clf.pt"))
print("DONE_CLF", flush=True)
