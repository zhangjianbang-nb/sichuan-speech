"""Measure how Sichuan-flavoured a TTS rendering is.

Two independent signals:
1. Acoustic statistics (F0 register/range, contour direction, speaking rate) compared
   against real Sichuan and real Mandarin anchors.
2. A KeSpeech-trained Southwestern-vs-Mandarin classifier. KeSpeech records both
   dialects in the same sessions, so the classifier is not learning corpus channel.
   It is applied to base and SFT TTS from the same vocoder, so any remaining
   synthetic-domain bias is constant and the base-vs-SFT gap is what matters.
"""
import os, sys, json, glob, re, argparse
import numpy as np
import soundfile as sf
import librosa

ap = argparse.ArgumentParser()
ap.add_argument("--sft_dir", default=os.path.expanduser("~/chuan/models/cosyvoice2-sichuan"))
ap.add_argument("--sft_spk", default="sichuan_female")
ap.add_argument("--out", default=os.path.expanduser("~/chuan/eval/accent"))
ap.add_argument("--clf", default=os.path.expanduser("~/chuan/work/clf_sw.pt"))
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("/home/whu/CosyVoice")
sys.path.append("/home/whu/CosyVoice/third_party/Matcha-TTS")
os.makedirs(args.out, exist_ok=True)

SICHUAN = [
    "今天天气好安逸哦，我们出去耍一哈嘛。",
    "你吃饭没有得？没得的话我们一起切吃火锅。",
    "这个事情我晓得，你莫着急，慢慢来。",
    "老板儿，来二两小面，多放点海椒。",
    "你娃儿最近在忙啥子嘛？",
    "昨晚上那个雨下得才大哦，裤子都打湿完了。",
    "这个菜巴适得板，再给我舀一碗饭。",
    "莫慌，等我把这个事情搞归一了再说。",
    "他屋头那个猫儿又跑到房顶上去了。",
    "我们是四川人，说话就是这个味道。",
]


def feats(path):
    y, sr = librosa.load(path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=30)
    f0, vflag, _ = librosa.pyin(y, fmin=70, fmax=450, sr=sr, frame_length=1024, hop_length=160)
    v = f0[~np.isnan(f0)]
    if len(v) < 10:
        return None
    semi = 12 * np.log2(v / 100.0)
    d = np.diff(semi)
    d = d[np.abs(d) > 0.02]
    # speaking rate proxy: energy-envelope peaks per second
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=160)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=160, units="time")
    return {
        "dur": round(float(len(y) / sr), 2),
        "f0_med_semi": round(float(np.median(semi)), 2),
        "f0_p10_semi": round(float(np.percentile(semi, 10)), 2),
        "f0_p90_semi": round(float(np.percentile(semi, 90)), 2),
        "f0_range": round(float(np.percentile(semi, 90) - np.percentile(semi, 10)), 2),
        "f0_std": round(float(np.std(semi)), 2),
        "fall_frac": round(float((d < 0).mean()), 3),
        "rate_syl_s": round(float(len(onsets) / max(1e-3, len(y) / sr)), 2),
    }


def agg(rows):
    rows = [r for r in rows if r]
    out = {}
    for k in rows[0]:
        out[k] = round(float(np.mean([r[k] for r in rows])), 3)
    out["n"] = len(rows)
    return out


# ---------- classifier ----------
def mel(p, n_mels=64, sr=16000, max_frames=400):
    y, _ = librosa.load(p, sr=sr)
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=160, n_mels=n_mels)
    m = librosa.power_to_db(m, ref=np.max)
    m = (m + 40.0) / 40.0
    if m.shape[1] > max_frames:
        m = m[:, :max_frames]
    else:
        m = np.pad(m, ((0, 0), (0, max_frames - m.shape[1])), constant_values=-1.0)
    return m.astype(np.float32)


def score_clf(paths):
    if not os.path.exists(args.clf):
        return None
    import torch, torch.nn as nn

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
            x = self.c(x)
            return self.f(x.flatten(1))

    net = Net()
    net.load_state_dict(torch.load(args.clf, map_location="cpu", weights_only=True))
    net.eval()
    ps = []
    with torch.no_grad():
        for p in paths:
            x = torch.from_numpy(mel(p)).unsqueeze(0).unsqueeze(0)
            ps.append(float(torch.softmax(net(x), -1)[0, 1]))
    return round(float(np.mean(ps)), 3) if ps else None


rows_out = {}
# anchors
sic = [r[1] for r in json.load(open(os.path.expanduser("~/chuan/data/sft/all_sichuan.json")))[:40]]
man = [r["wav"] for r in json.load(open(os.path.expanduser("~/chuan/data/eval/kespeech_mandarin.json")))[:40]]
rows_out["anchor_real_sichuan"] = agg([feats(p) for p in sic])
rows_out["anchor_real_sichuan"]["p_sw"] = score_clf(sic)
rows_out["anchor_real_mandarin"] = agg([feats(p) for p in man])
rows_out["anchor_real_mandarin"]["p_sw"] = score_clf(man)

from cosyvoice.cli.cosyvoice import AutoModel as CModel
import torch

A = "/home/whu/CosyVoice/asset/zero_shot_prompt.wav"
PT = "希望你以后能够做的比我还好呦。"


def gen(model, sentences, mode, sub, spk=None, prompt=None):
    d = os.path.join(args.out, sub)
    os.makedirs(d, exist_ok=True)
    paths = []
    for i, t in enumerate(sentences):
        it = (model.inference_zero_shot(t, prompt[1], prompt[0], stream=False)
              if mode == "zs" else model.inference_sft(t, spk, stream=False))
        for j, o in enumerate(it):
            p = os.path.join(d, "%02d_%d.wav" % (i, j))
            sf.write(p, o["tts_speech"].squeeze().cpu().numpy(), model.sample_rate)
            paths.append(p)
    return paths


base = CModel(model_dir="/home/whu/models/CosyVoice2-0.5B", fp16=True)
p_mand = gen(base, SICHUAN, "zs", "base_mandarin_prompt", prompt=(A, PT))
p_sich = gen(base, SICHUAN, "zs", "base_sichuan_prompt",
             prompt=("/home/whu/chuan/data/raw/magic_tts/1776909066881_g4plqjyx_32946.wav",
                     "领导想安排我转岗，我礼貌拒绝，我更想在现在的岗位发展。"))
del base
torch.cuda.empty_cache()

sft = CModel(model_dir=args.sft_dir, fp16=True)
p_sft = gen(sft, SICHUAN, "sft", "sft", spk=args.sft_spk)
del sft

for name, ps in [("base_mandarin_prompt", p_mand), ("base_sichuan_prompt", p_sich), ("sft", p_sft)]:
    rows_out[name] = agg([feats(p) for p in ps])
    rows_out[name]["p_sw"] = score_clf(ps)

json.dump(rows_out, open(os.path.join(args.out, "accent_report.json"), "w"),
          ensure_ascii=False, indent=1)
keys = ["f0_med_semi", "f0_range", "f0_std", "fall_frac", "rate_syl_s", "p_sw", "dur"]
print("%-24s %s" % ("set", "  ".join("%10s" % k for k in keys)))
for k, v in rows_out.items():
    print("%-24s %s" % (k, "  ".join("%10s" % v.get(x) for x in keys)))
print("DONE_ACCENT", flush=True)
