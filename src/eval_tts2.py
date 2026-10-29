"""Evaluate dialect transfer of TTS via ASR-CER contrast (channel-robust).

Anchors: real Sichuan audio vs real Mandarin audio, scored by the same ASR.
Then base TTS vs SFT TTS. Sichuan-accented speech is harder for a Mandarin-only
ASR, so a rise in CER vs. the base model is evidence of accent shift.
"""
import os, sys, json, glob, argparse
import numpy as np
import soundfile as sf
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--sft_dir", default=os.path.expanduser("~/chuan/models/cosyvoice2-sichuan"))
ap.add_argument("--sft_spk", default="sichuan_female")
ap.add_argument("--out", default=os.path.expanduser("~/chuan/eval/round"))
ap.add_argument("--asr_ckpt", default="")
ap.add_argument("--tag", default="sft")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("/home/whu/CosyVoice")
sys.path.append("/home/whu/CosyVoice/third_party/Matcha-TTS")

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
if args.quick:
    SICHUAN = SICHUAN[:3]
    MANDARIN = MANDARIN[:2]

os.makedirs(args.out, exist_ok=True)


def pick_prompt():
    txt = {}
    for line in open(os.path.expanduser("~/chuan/data/raw/magic_tts/txt.txt")):
        p = line.strip().split("\t")
        if len(p) >= 2:
            txt[p[0]] = p[1]
    best = None
    for f in glob.glob(os.path.expanduser("~/chuan/data/raw/magic_tts/*.wav")):
        d = sf.info(f).duration
        b = os.path.basename(f)
        if 4.5 <= d <= 9 and b in txt:
            if best is None or d < best[2]:
                best = (f, txt[b], d)
    return best


def gen(model, sentences, mode, outdir, spk=None, prompt=None):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for i, t in enumerate(sentences):
        outs = []
        it = (model.inference_zero_shot(t, prompt[1], prompt[0], stream=False)
              if mode == "zero_shot" else model.inference_sft(t, spk, stream=False))
        for j, o in enumerate(it):
            p = os.path.join(outdir, "%02d_%d.wav" % (i, j))
            sf.write(p, o["tts_speech"].squeeze().cpu().numpy(), model.sample_rate)
            outs.append(p)
        paths.append(outs)
    return paths


# ---------- ASR scorers ----------
import re
from funasr import AutoModel as FModel
base_asr = FModel(model="/home/whu/models/SenseVoiceSmall", device="cpu", disable_update=True)

ft_asr = None
if args.asr_ckpt:
    import train_asr_lib as tal
    ft_asr = tal.load_finetuned(args.asr_ckpt, device="cpu")


def asr_text(path, which="base"):
    if which == "base":
        r = base_asr.generate(input=path, cache={}, language="zh", use_itn=True, batch_size_s=60)
        return re.sub(r"<\|[^|]*\|>", "", r[0]["text"]).strip()
    return ft_asr.transcribe(path)


def cer(ref, hyp):
    r, h = list(ref), list(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return int(d[len(r), len(h)]), len(r)


def score(paths, refs, name):
    rows, tot_e, tot_n = [], 0, 0
    for i, outs in enumerate(paths):
        for p in outs:
            hyp = asr_text(p, "base")
            e, n = cer(refs[i], hyp)
            tot_e += e
            tot_n += n
            rows.append({"wav": p, "hyp": hyp, "cer": round(e / max(1, n), 4)})
    r = {"mean_cer": round(tot_e / max(1, tot_n), 4), "rows": rows}
    print("%-32s CER %.4f" % (name, r["mean_cer"]), flush=True)
    return r


report = {}
prompt = pick_prompt()
print("prompt:", prompt[0], round(prompt[2], 2), prompt[1], flush=True)

# anchors: real speech
sic_ref = [r[2] for r in json.load(open(os.path.expanduser("~/chuan/data/sft/all_sichuan.json")))[:40]]
sic_wav = [r[1] for r in json.load(open(os.path.expanduser("~/chuan/data/sft/all_sichuan.json")))[:40]]
report["anchor_real_sichuan"] = score([[w] for w in sic_wav], sic_ref, "anchor_real_sichuan")

man_rows = json.load(open(os.path.expanduser("~/chuan/data/eval/kespeech_mandarin.json")))[:40]
report["anchor_real_mandarin"] = score([[r["wav"]] for r in man_rows],
                                      [r["text"] for r in man_rows], "anchor_real_mandarin")

from cosyvoice.cli.cosyvoice import AutoModel as CModel
base = CModel(model_dir="/home/whu/models/CosyVoice2-0.5B", fp16=True)
p_base = gen(base, SICHUAN, "zero_shot", os.path.join(args.out, "base_sichuan"), prompt=prompt)
del base
torch.cuda.empty_cache()

sft = CModel(model_dir=args.sft_dir, fp16=True)
p_sft = gen(sft, SICHUAN, "sft", os.path.join(args.out, "sft_sichuan"), spk=args.sft_spk)
p_sft_man = gen(sft, MANDARIN, "sft", os.path.join(args.out, "sft_mandarin"), spk=args.sft_spk)
del sft
torch.cuda.empty_cache()

report["base_zero_shot_sichuan"] = score(p_base, SICHUAN, "base_zero_shot_sichuan")
report["sft_sichuan"] = score(p_sft, SICHUAN, "sft_sichuan")
report["sft_mandarin"] = score(p_sft_man, MANDARIN, "sft_mandarin")

json.dump(report, open(os.path.join(args.out, "report_%s.json" % args.tag), "w"),
          ensure_ascii=False, indent=1)
print("DONE_EVAL", flush=True)
