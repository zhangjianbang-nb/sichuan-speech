"""Final TTS evaluation: intelligibility (ASR CER) + acoustic dialect statistics.

Sets compared, all from the same vocoder so any synthetic-domain bias cancels:
  base_mandarin_prompt  - stock CosyVoice2 + official Mandarin prompt
  base_sichuan_prompt   - stock CosyVoice2 + Sichuan prompt (accent cloned from prompt)
  sft_female            - Sichuan-SFT LLM + MagicHub Chengdu female speaker embedding
  sft_avg               - Sichuan-SFT LLM + mean embedding of KeSpeech SW speakers
Anchors: 40 real Sichuan utts and 40 real Mandarin utts, for reference ranges.
"""
import os, sys, json, glob, re, argparse
import numpy as np
import soundfile as sf
import librosa
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--sft_dir", default=os.path.expanduser("~/chuan/models/cosyvoice2-sichuan"))
ap.add_argument("--out", default=os.path.expanduser("~/chuan/eval/final"))
ap.add_argument("--n_anchor", type=int, default=40)
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
MANDARIN = [
    "今天天气很好，我们出去走一走吧。",
    "你吃饭了吗？没有的话我们一起去吃火锅。",
    "这件事情我知道，你不要着急，慢慢来。",
    "老板，来二两小面，多放一点辣椒。",
]

SICH_PROMPT = (os.path.expanduser("~/chuan/data/raw/magic_tts/1776909066881_g4plqjyx_32946.wav"),
               "领导想安排我转岗，我礼貌拒绝，我更想在现在的岗位发展。")
MAND_PROMPT = ("/home/whu/CosyVoice/asset/zero_shot_prompt.wav", "希望你以后能够做的比我还好呦。")


def norm(t):
    return re.sub(r"[\s，。、？！；：,.?!;:\"'“”‘’（）()《》<>·—\-]", "", re.sub(r"<\|[^|]*\|>", "", t))


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return int(d[len(r), len(h)]), len(r)


def feats(path):
    y, sr = librosa.load(path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < sr * 0.3:
        return None
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=450, sr=sr, frame_length=1024, hop_length=160)
    v = f0[~np.isnan(f0)]
    if len(v) < 10:
        return None
    semi = 12 * np.log2(v / 100.0)
    d = np.diff(semi)
    d = d[np.abs(d) > 0.02]
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=160)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=160, units="time")
    return {"dur": round(float(len(y) / sr), 2),
            "f0_med_semi": round(float(np.median(semi)), 2),
            "f0_range": round(float(np.percentile(semi, 90) - np.percentile(semi, 10)), 2),
            "f0_std": round(float(np.std(semi)), 2),
            "fall_frac": round(float((d < 0).mean()), 3),
            "rate_syl_s": round(float(len(onsets) / max(1e-3, len(y) / sr)), 2)}


def agg(rows):
    rows = [r for r in rows if r]
    if not rows:
        return {}
    out = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}
    out["n"] = len(rows)
    return out


from funasr import AutoModel as F
asr = F(model="/home/whu/models/SenseVoiceSmall", device="cpu", disable_update=True)


def hyp(p):
    r = asr.generate(input=p, cache={}, language="zh", use_itn=False, batch_size_s=60)
    return re.sub(r"<\|[^|]*\|>", "", r[0]["text"]).strip()


def score_set(paths, refs, name, store):
    te = tn = 0
    rows = []
    for i, outs in enumerate(paths):
        for p in outs:
            h = hyp(p)
            e, n = cer(refs[i], h)
            te += e
            tn += n
            rows.append({"wav": p, "ref": refs[i], "hyp": h, "cer": round(e / max(1, n), 4)})
    flat = [p for outs in store for p in outs]
    r = {"mean_cer": round(te / max(1, tn), 4), "acoustic": agg([feats(p) for p in flat]),
         "rows": rows}
    print("%-22s CER %.4f  f0_med %s  rate %s  fall %s" %
          (name, r["mean_cer"], r["acoustic"].get("f0_med_semi"),
           r["acoustic"].get("rate_syl_s"), r["acoustic"].get("fall_frac")), flush=True)
    return r


def gen(model, sentences, mode, sub, spk=None, prompt=None):
    d = os.path.join(args.out, sub)
    os.makedirs(d, exist_ok=True)
    paths = []
    for i, t in enumerate(sentences):
        it = (model.inference_zero_shot(t, prompt[1], prompt[0], stream=False)
              if mode == "zs" else model.inference_sft(t, spk, stream=False))
        outs = []
        for j, o in enumerate(it):
            p = os.path.join(d, "%02d_%d.wav" % (i, j))
            sf.write(p, o["tts_speech"].squeeze().cpu().numpy(), model.sample_rate)
            outs.append(p)
        paths.append(outs)
    return paths


rep = {}
sic = [r[1] for r in json.load(open(os.path.expanduser("~/chuan/data/sft/all_sichuan.json")))[:args.n_anchor]]
sic_ref = [r[2] for r in json.load(open(os.path.expanduser("~/chuan/data/sft/all_sichuan.json")))[:args.n_anchor]]
man = json.load(open(os.path.expanduser("~/chuan/data/eval/kespeech_mandarin.json")))[:args.n_anchor]
rep["anchor_real_sichuan"] = {"acoustic": agg([feats(p) for p in sic]),
                              "mean_cer": round(sum(cer(t, hyp(p))[0] for p, t in zip(sic, sic_ref)) /
                                                max(1, sum(len(norm(t)) for t in sic_ref)), 4)}
rep["anchor_real_mandarin"] = {"acoustic": agg([feats(p) for p in [r["wav"] for r in man]]),
                               "mean_cer": round(sum(cer(r["text"], hyp(r["wav"]))[0] for r in man) /
                                                 max(1, sum(len(norm(r["text"])) for r in man)), 4)}
print("anchors done", flush=True)

from cosyvoice.cli.cosyvoice import AutoModel as CModel
base = CModel(model_dir="/home/whu/models/CosyVoice2-0.5B", fp16=True)
p_mand = gen(base, SICHUAN, "zs", "base_mandarin_prompt", prompt=MAND_PROMPT)
p_sich = gen(base, SICHUAN, "zs", "base_sichuan_prompt", prompt=SICH_PROMPT)
del base
torch.cuda.empty_cache()

sft = CModel(model_dir=args.sft_dir, fp16=True)
avail = sft.list_available_spks()
print("sft spks:", avail, flush=True)
p_f = gen(sft, SICHUAN, "sft", "sft_female", spk="sichuan_female")
p_a = gen(sft, SICHUAN, "sft", "sft_avg", spk="sichuan_avg") if "sichuan_avg" in avail else None
p_man = gen(sft, MANDARIN, "sft", "sft_mandarin", spk="sichuan_female")
del sft

rep["base_mandarin_prompt"] = score_set(p_mand, SICHUAN, "base_mandarin_prompt", p_mand)
rep["base_sichuan_prompt"] = score_set(p_sich, SICHUAN, "base_sichuan_prompt", p_sich)
rep["sft_female"] = score_set(p_f, SICHUAN, "sft_female", p_f)
if p_a:
    rep["sft_avg"] = score_set(p_a, SICHUAN, "sft_avg", p_a)
rep["sft_mandarin_regression"] = score_set(p_man, MANDARIN, "sft_mandarin_regression", p_man)

json.dump(rep, open(os.path.join(args.out, "tts_eval.json"), "w"), ensure_ascii=False, indent=1)
print("DONE_TTS_EVAL", flush=True)
