"""Check that generated audio actually carries the requested speaker identity.

Extracts campplus embeddings from the generated wavs and reports cosine similarity to the
speaker embedding used at synthesis. Low similarity means the flow ignored the embedding
and the voice is not the intended one.
"""
import os, sys, json, glob
import numpy as np
import soundfile as sf
import torch
import onnxruntime

M = "/home/whu/models/CosyVoice2-0.5B"
OUT = os.path.expanduser("~/chuan/eval/final")

opt = onnxruntime.SessionOptions()
sess = onnxruntime.InferenceSession(os.path.join(M, "campplus.onnx"), sess_options=opt,
                                    providers=["CPUExecutionProvider"])


def emb(path):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 16000:
        import torchaudio
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, 16000).numpy()
    f = torch.from_numpy(w).unsqueeze(0)
    feat = torch.log(torch.stft(f, n_fft=400, hop_length=160, win_length=400,
                                window=torch.hann_window(400), center=False,
                                return_complex=True).abs() + 1e-6)
    # CosyVoice frontend uses kaldi fbank; approximate with mel filterbank
    import torchaudio.compliance.kaldi as kaldi
    fb = kaldi.fbank(f, num_mel_bins=80, sample_frequency=16000, dither=0.0)
    fb = fb - fb.mean(0, keepdim=True)
    out = sess.run(None, {sess.get_inputs()[0].name: fb.unsqueeze(0).numpy()})[0]
    v = torch.from_numpy(out).reshape(1, -1)
    return torch.nn.functional.normalize(v, dim=1)


def target(spk):
    d = torch.load(os.path.expanduser("~/chuan/models/cosyvoice2-sichuan2/spk2info.pt"),
                   map_location="cpu", weights_only=False)
    return torch.nn.functional.normalize(d[spk]["embedding"].reshape(1, -1), dim=1)


res = {}
for sub, spk in [("sft_female", "sichuan_female"), ("sft_avg", "sichuan_avg")]:
    paths = sorted(glob.glob(os.path.join(OUT, sub, "*.wav")))
    if not paths:
        continue
    t = target(spk)
    sims = []
    for p in paths:
        try:
            sims.append(float((emb(p) * t).sum()))
        except Exception as e:
            print("skip", p, repr(e)[:60])
    res[sub] = {"n": len(sims), "mean_cos": round(float(np.mean(sims)), 3),
                "min_cos": round(float(np.min(sims)), 3)}
    print(sub, res[sub], flush=True)

# reference: similarity between the prompt wav and the sichuan_female embedding
try:
    p = os.path.expanduser("~/chuan/data/raw/magic_tts/1776909066881_g4plqjyx_32946.wav")
    res["prompt_self"] = {"cos": round(float((emb(p) * target("sichuan_female")).sum()), 3)}
    print("prompt_self", res["prompt_self"])
except Exception as e:
    print("prompt_self failed", repr(e)[:80])

json.dump(res, open(os.path.join(OUT, "speaker_sim.json"), "w"), indent=1)
print("DONE_SPKSIM", flush=True)
