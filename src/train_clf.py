"""Southwestern-vs-Mandarin classifier trained on KeSpeech (channel-matched dialects).

Speaker-disjoint CV so the reported accuracy is not speaker memorisation. Applied to
TTS output the absolute score is domain-shifted, but base and SFT share the same
vocoder, so their difference is meaningful.
"""
import os, json, sys
import numpy as np
import librosa
import torch
import torch.nn as nn

E = os.path.expanduser("~/chuan/data/eval")
OUT = os.path.expanduser("~/chuan/work/clf_sw.pt")
N_MELS, MAX_FRAMES = 64, 400


def mel(p):
    y, _ = librosa.load(p, sr=16000)
    m = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=160, n_mels=N_MELS)
    m = librosa.power_to_db(m, ref=np.max)
    m = (m + 40.0) / 40.0
    if m.shape[1] > MAX_FRAMES:
        m = m[:, :MAX_FRAMES]
    else:
        m = np.pad(m, ((0, 0), (0, MAX_FRAMES - m.shape[1])), constant_values=-1.0)
    return m.astype(np.float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)))
        self.f = nn.Linear(64 * 4 * 8, 2)

    def forward(self, x):
        return self.f(self.c(x).flatten(1))


def spk(p):
    return os.path.basename(p).split("_")[0]


def main():
    rows = [(r["wav"], 1) for r in json.load(open(os.path.join(E, "kespeech_sw.json")))]
    rows += [(r["wav"], 0) for r in json.load(open(os.path.join(E, "kespeech_mandarin.json")))]
    X = np.stack([mel(p) for p, _ in rows])[:, None]
    y = np.array([l for _, l in rows])
    g = np.array([spk(p) for p, _ in rows])
    print("samples", len(y), "speakers", len(set(g)), "sw speakers", len({s for s, l in zip(g, y) if l == 1}))

    from sklearn.model_selection import GroupKFold
    accs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        net = Net()
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        lossf = nn.CrossEntropyLoss()
        xt = torch.from_numpy(X[tr])
        yt = torch.from_numpy(y[tr])
        for ep in range(12):
            net.train()
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 32):
                idx = perm[i:i + 32]
                opt.zero_grad()
                loss = lossf(net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(torch.from_numpy(X[te])).argmax(-1).numpy()
        acc = float((pred == y[te]).mean())
        accs.append(acc)
        print("fold acc %.3f (n=%d)" % (acc, len(te)), flush=True)
    print("speaker-disjoint CV acc %.3f +- %.3f" % (float(np.mean(accs)), float(np.std(accs))))

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss()
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    for ep in range(20):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = lossf(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    torch.save(net.state_dict(), OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
