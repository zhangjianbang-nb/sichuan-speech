"""Cross-ASR dialect check for the synthesized TTS audio.

Two recognizers score the same audio:
  base  - stock SenseVoiceSmall (Mandarin-centric)
  ft    - Sichuan-finetuned SenseVoiceSmall (exp/asr2/epoch_2.pt)

Interpretation:
  * both low      -> standard Mandarin reading, no dialect effect
  * ft << base    -> dialect phonology that the dialect-adapted recognizer handles
  * both high     -> garbled audio, not a dialect
This separates "authentic accent" from "unintelligible garbage", which a single
CER number cannot do.
"""
import os, sys, json, re, glob
import numpy as np
import torch

sys.path.append("/home/whu/chuan/work")
import train_asr_lib as tal
from funasr import AutoModel as F

OUT = os.path.expanduser("~/chuan/eval/final")
FT = os.path.expanduser("~/chuan/exp/asr2/epoch_2.pt")

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
MANDARIN = [
    "今天天气很好，我们出去走一走吧。",
    "你吃饭了吗？没有的话我们一起去吃火锅。",
    "这件事情我知道，你不要着急，慢慢来。",
    "老板，来二两小面，多放一点辣椒。",
]


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


base = F(model="/home/whu/models/SenseVoiceSmall", device="cpu", disable_update=True)
ft = tal.load_finetuned(FT, device="cpu")


def base_tr(p):
    return re.sub(r"<\|[^|]*\|>", "", base.generate(input=p, cache={}, language="zh",
                                                    use_itn=False, batch_size_s=60)[0]["text"])


sets = []
for sub in ["base_mandarin_prompt", "base_sichuan_prompt", "sft_female", "sft_avg", "sft_mandarin"]:
    d = os.path.join(OUT, sub)
    if os.path.isdir(d):
        sets.append(sub)

rep = {}
for sub in sets:
    refs = MANDARIN if sub == "sft_mandarin" else SICHUAN
    files = sorted(glob.glob(os.path.join(OUT, sub, "*.wav")))
    eb = nb = ef = nf = 0
    rows = []
    for p in files:
        idx = int(os.path.basename(p).split("_")[0])
        ref = refs[idx]
        hb = base_tr(p)
        hf = ft.transcribe(p)
        a, n1 = cer(ref, hb)
        b, n2 = cer(ref, hf)
        eb += a
        nb += n1
        ef += b
        nf += n2
        rows.append({"wav": os.path.basename(p), "ref": ref, "base": hb, "ft": hf,
                     "cer_base": round(a / max(1, n1), 4), "cer_ft": round(b / max(1, n2), 4)})
    cb, cf = round(eb / max(1, nb), 4), round(ef / max(1, nf), 4)
    rep[sub] = {"n": len(files), "cer_base": cb, "cer_ft": cf,
                "delta": round(cf - cb, 4), "rows": rows}
    print("%-22s base %.4f  ft %.4f  delta %+.4f" % (sub, cb, cf, cf - cb), flush=True)

json.dump(rep, open(os.path.join(OUT, "cross_asr.json"), "w"), ensure_ascii=False, indent=1)
print("DONE_CROSS", flush=True)
