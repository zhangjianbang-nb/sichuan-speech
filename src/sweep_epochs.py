"""Sweep SFT checkpoints and score them on the accent objective, not on CV loss.

The training CV bottoms out at epoch 4, but CV measures token-prediction loss, not accent
strength. This script generates the same sentences with every checkpoint under a fixed seed
and reports, per checkpoint:
  cer_base  - CER of the stock Mandarin ASR (rises as pronunciation drifts from Mandarin)
  cer_ft    - CER of the Sichuan-finetuned ASR (stays low only if the audio is intelligible)
  rate      - syllables/second from onset detection (real Sichuan anchor: 4.67)
The accent-optimal checkpoint keeps cer_ft low while cer_base approaches the real-Sichuan
anchor value (~0.077).
"""
import os, sys, json, re, glob, argparse
import numpy as np
import soundfile as sf
import librosa
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", default=os.path.expanduser("~/chuan/models/cosyvoice2-sichuan2"))
ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/chuan/exp/llm2"))
ap.add_argument("--out", default=os.path.expanduser("~/chuan/eval/sweep"))
ap.add_argument("--spk", default="sichuan_female")
ap.add_argument("--n_sent", type=int, default=20)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("/home/whu/CosyVoice")
sys.path.append("/home/whu/CosyVoice/third_party/Matcha-TTS")
sys.path.append("/home/whu/chuan/work")
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
    "你莫要扯把子，这个事情我清楚得很。",
    "走嘛，我们切河边头转一圈，顺便买点卤菜。",
    "那个妹儿长得乖得很，说话也温柔。",
    "天都黑了，你还不回来吃饭啊？",
    "我跟你讲，这个事情办得巴适。",
    "你不要扭到我费，我自己晓得咋个办。",
    "来嘛，坐下喝口茶，慢慢摆龙门阵。",
    "这个瓜娃子，一天到黑就晓得耍。",
    "我屋头的老汉儿今年七十多了，身体还硬朗。",
    "四川的火锅嘛，就是要吃辣才安逸。",
][:args.n_sent]

import train_asr_lib as tal
from funasr import AutoModel as F

base = F(model="/home/whu/models/SenseVoiceSmall", device="cpu", disable_update=True)
ft = tal.load_finetuned(os.path.expanduser("~/chuan/exp/asr2/epoch_2.pt"), device="cpu")


def norm(t):
    t = re.sub(r"<\|[^|]*\|>", "", t)
    return re.sub(r"[\s，。、？！；：,.?!;:\"'“”‘’（）()《》<>·—\-]", "", t)


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return int(d[len(r), len(h)]), len(r)


def ac(p):
    y, sr = librosa.load(p, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=30)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=160)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=160, units="time")
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=450, sr=sr, frame_length=1024, hop_length=160)
    v = f0[~np.isnan(f0)]
    semi = 12 * np.log2(v / 100.0) if len(v) else np.array([np.nan])
    return {"dur": round(float(len(y) / sr), 2),
            "rate": round(float(len(on) / max(1e-3, len(y) / sr)), 2),
            "f0_med": round(float(np.nanmedian(semi)), 2)}


from cosyvoice.cli.cosyvoice import AutoModel as CModel
m = CModel(model_dir=args.model_dir, fp16=True)
print("spks", m.list_available_spks(), flush=True)

ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "epoch_*_whole.pt")),
               key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))
ckpts = [("base", None)] + [(os.path.basename(c).replace("_whole.pt", ""), c) for c in ckpts]

rep = {}
for tag, ck in ckpts:
    if ck is None:
        src = "/home/whu/models/CosyVoice2-0.5B/llm.pt"
    else:
        src = ck
    sd = torch.load(src, map_location="cpu", weights_only=True)
    sd.pop("epoch", None)
    sd.pop("step", None)
    m.model.llm.load_state_dict(sd, strict=True)
    m.model.llm.eval()
    d = os.path.join(args.out, tag)
    os.makedirs(d, exist_ok=True)
    eb = nb = ef = nf = 0
    acs = []
    for i, t in enumerate(SICHUAN):
        torch.manual_seed(1000 + i)
        np.random.seed(1000 + i)
        outs = [o["tts_speech"].squeeze().cpu().numpy()
                for o in m.inference_sft(t, args.spk, stream=False)]
        wav = np.concatenate(outs)
        p = os.path.join(d, "%02d.wav" % i)
        sf.write(p, wav, m.sample_rate)
        hb = re.sub(r"<\|[^|]*\|>", "", base.generate(input=p, cache={}, language="zh",
                                                     use_itn=False, batch_size_s=60)[0]["text"])
        hf = ft.transcribe(p)
        a, n1 = cer(t, hb)
        b, n2 = cer(t, hf)
        eb += a
        nb += n1
        ef += b
        nf += n2
        acs.append(ac(p))
    cb, cf = round(eb / max(1, nb), 4), round(ef / max(1, nf), 4)
    rate = round(float(np.mean([x["rate"] for x in acs])), 2)
    rep[tag] = {"cer_base": cb, "cer_ft": cf, "rate": rate,
                "dur": round(float(np.mean([x["dur"] for x in acs])), 2)}
    print("%-8s cer_base %.4f  cer_ft %.4f  rate %.2f  dur %.2f" %
          (tag, cb, cf, rate, rep[tag]["dur"]), flush=True)
    json.dump(rep, open(os.path.join(args.out, "sweep.json"), "w"), indent=1)

print("DONE_SWEEP", flush=True)
