"""Shared architecture for the Sichuan-vs-Mandarin dialect classifier."""
import numpy as np
import torch
import torch.nn as nn

SR = 16000
N_MELS = 64
HOP = 160
WIN = 400


def mel(wav):
    import torchaudio
    w = torch.from_numpy(np.asarray(wav)).float()
    m = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=WIN, hop_length=HOP, n_mels=N_MELS, f_min=0, f_max=8000)(w)
    return torch.log(m.clamp(min=1e-6))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [1, 32, 64, 128, 128]
        blocks = []
        for i in range(len(ch) - 1):
            blocks += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1),
                       nn.BatchNorm2d(ch[i + 1]), nn.ReLU(),
                       nn.MaxPool2d(2)]
        self.cnn = nn.Sequential(*blocks)
        self.gru = nn.GRU(128 * (N_MELS // 16), 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))      # (B, C, F', T')
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        return self.head(x)
