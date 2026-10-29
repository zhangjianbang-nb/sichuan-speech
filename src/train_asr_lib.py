"""Load a fine-tuned SenseVoice checkpoint and transcribe / score CER."""
import os, re
import numpy as np
import soundfile as sf
import torch

SPECIAL_ZH = 24884
SPECIAL_SPEECH = 24993
SPECIAL_NOITN = 25017
VOCAB_SPECIAL_START = 24884


class FTAsr:
    def __init__(self, ckpt, device="cpu"):
        from funasr import AutoModel
        from funasr.frontends.wav_frontend import WavFrontend
        am = AutoModel(model="/home/whu/models/SenseVoiceSmall", device="cpu", disable_update=True)
        self.model = am.model
        self.tok = am.kwargs["tokenizer"]
        self.fe = WavFrontend(**am.kwargs["frontend_conf"])
        if ckpt:
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
            self.model.load_state_dict(sd, strict=False)
        self.model.to(device).eval()
        self.device = device

    def _load(self, path):
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, 16000).numpy()
        return wav

    @torch.no_grad()
    def transcribe(self, path):
        wav = self._load(path)
        w = torch.from_numpy(wav).unsqueeze(0)
        f, fl = self.fe(w, torch.tensor([wav.shape[0]]))
        f = f.to(self.device)
        fl = fl.to(self.device)
        text = torch.tensor([[SPECIAL_ZH, SPECIAL_SPEECH, 0, SPECIAL_NOITN]], dtype=torch.long).to(self.device)
        enc, enc_lens = self.model.encode(f, fl, text)
        logits = self.model.ctc.log_softmax(enc)
        ids = logits[0, :int(enc_lens[0])].argmax(dim=-1).tolist()
        prev, res = None, []
        for i in ids:
            if i != prev:
                if i != 0 and i < VOCAB_SPECIAL_START:
                    res.append(i)
                prev = i
        return self.tok.decode(res)

    def score(self, rows):
        tot_e = tot_n = 0
        out = []
        for wav, ref in rows:
            hyp = self.transcribe(wav)
            e, n = _cer(ref, hyp)
            tot_e += e
            tot_n += n
            out.append({"wav": wav, "ref": ref, "hyp": hyp, "cer": round(e / max(1, n), 4)})
        return tot_e / max(1, tot_n), out


def _cer(ref, hyp):
    r, h = list(ref), list(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return int(d[len(r), len(h)]), len(r)


def load_finetuned(ckpt, device="cpu"):
    return FTAsr(ckpt, device)


def load_base(device="cpu"):
    """Stock SenseVoiceSmall, no fine-tune applied. Used as the second opinion in cross-ASR."""
    return FTAsr(None, device)
