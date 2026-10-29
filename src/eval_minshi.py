"""Objective Sichuan-accent test: the s/sh (平翘舌) merger.

Sichuan dialect merges the retroflex series zh/ch/sh into the alveolar z/c/s, so 「四」(si)
and 「十」(shi) become acoustically near-identical. Standard Mandarin keeps them apart, and
the retroflex fricative has a distinctly lower spectral centroid.

Method: CTC forced alignment against the known transcript, take the onset window of every
character whose initial is s- or sh-, and measure the spectral centroid of the fricative.
Report centroid(s) - centroid(sh):
  large gap  -> Mandarin-style distinction (no Sichuan accent)
  ~zero gap  -> Sichuan-style merger

Calibrated against real Sichuan and real Mandarin recordings from KeSpeech, so the target
range comes from human speech, not from an assumption.
"""
import os, sys, json, re, glob, argparse
import numpy as np
import soundfile as sf
import torch
import torchaudio

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/chuan/eval/final/minshi"))
ap.add_argument("--per_set", type=int, default=25)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("/home/whu/CosyVoice")
sys.path.append("/home/whu/CosyVoice/third_party/Matcha-TTS")
sys.path.append("/home/whu/chuan/work")
os.makedirs(args.out, exist_ok=True)

import train_asr_lib as tal

S_CHARS = set("四三苏送散僧搜塞丧森")
SH_CHARS = set("十是时说上山水书少收深")

ft = tal.load_finetuned(os.path.expanduser("~/chuan/exp/asr2/epoch_2.pt"), device="cpu")
model = ft.model
tok = ft.tok
fe = ft.fe
print("tokenizer check:", [(c, tok.encode(c)) for c in "四十是"], flush=True)


def wav_of(p):
    w, sr = sf.read(p, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 16000:
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, 16000).numpy()
    return w


def centroid(seg):
    seg = seg * np.hanning(len(seg))
    sp = np.abs(np.fft.rfft(seg, n=2048))
    f = np.fft.rfftfreq(2048, 1 / 16000)
    lo = (f > 1000) & (f < 8000)
    if sp[lo].sum() < 1e-9:
        return np.nan
    return float((sp[lo] * f[lo]).sum() / sp[lo].sum())


def analyse(p, text):
    w = wav_of(p)
    dur = len(w) / 16000
    f, fl = fe(torch.from_numpy(w).unsqueeze(0), torch.tensor([len(w)]))
    ids = tok.encode(text)
    tb = torch.tensor([[24884, 24993, 0, 25017]], dtype=torch.long)
    with torch.no_grad():
        enc, enc_lens = model.encode(f, fl, tb)
        logits = model.ctc.log_softmax(enc)
    T = int(enc_lens[0])
    hop = dur / T
    lp = logits[0, :T].unsqueeze(0).float()
    tg = torch.tensor([ids], dtype=torch.long)
    try:
        al, _ = torchaudio.functional.forced_align(lp, tg, torch.tensor([T]), torch.tensor([len(ids)]), blank=0)
    except Exception as e:
        return None, repr(e)[:80]
    al = al[0].tolist()
    # forced_align returns label ids, not target positions. Group frames into runs and
    # walk the runs and targets in lockstep so repeated characters get distinct spans.
    runs = []
    prev = None
    for t, a in enumerate(al):
        if a == 0:
            prev = None
            continue
        if prev is None or a != prev:
            runs.append([a, t, t])
            prev = a
        else:
            runs[-1][2] = t
    spans = {}
    ptr = 0
    for lab, t0, t1 in runs:
        if ptr < len(ids) and ids[ptr] == lab:
            spans[ptr] = (t0, t1)
            ptr += 1
    s_c, sh_c = [], []
    for k, ch in enumerate(text):
        span = spans.get(k)
        if not span:
            continue
        # window = onset of the character, capped so it stays inside the initial consonant
        t0, t1 = span
        n = max(1, t1 - t0 + 1)
        win = min(0.085, n * hop * 0.45)
        a = int(t0 * 16000 * hop)
        b = a + int(win * 16000)
        if b > len(w) or a >= len(w) or b - a < 160:
            continue
        c = centroid(w[a:b])
        if np.isnan(c):
            continue
        if ch in S_CHARS:
            s_c.append(c)
        elif ch in SH_CHARS:
            sh_c.append(c)
    return {"s": s_c, "sh": sh_c, "n_tok": len(ids), "T": T, "hop_ms": round(hop * 1000, 1)}, None


def collect(items, name):
    S, SH = [], []
    used = 0
    for p, t in items:
        r, err = analyse(p, t)
        if err or not r:
            continue
        if not r["s"] or not r["sh"]:
            continue
        used += 1
        S += r["s"]
        SH += r["sh"]
    if not S or not SH:
        print("%-18s no data" % name, flush=True)
        return None
    ms, msh = float(np.mean(S)), float(np.mean(SH))
    print("%-18s utts %2d  n_s %3d n_sh %3d  centroid_s %.0f  centroid_sh %.0f  gap %+.0f Hz" %
          (name, used, len(S), len(SH), ms, msh, ms - msh), flush=True)
    return {"utts": used, "n_s": len(S), "n_sh": len(SH),
            "centroid_s": round(ms, 1), "centroid_sh": round(msh, 1),
            "gap_hz": round(ms - msh, 1)}


rep = {}


def pick(json_path, per):
    rows = json.load(open(os.path.expanduser(json_path)))
    out = []
    for r in rows:
        t = r["text"]
        if any(c in S_CHARS for c in t) and any(c in SH_CHARS for c in t) and 1.0 < len(t) < 30:
            out.append((r["wav"], t))
        if len(out) >= per:
            break
    return out


rep["real_sichuan"] = collect(pick("~/chuan/data/eval/kespeech_sw.json", args.per_set), "real_sichuan")
rep["real_mandarin"] = collect(pick("~/chuan/data/eval/kespeech_mandarin.json", args.per_set), "real_mandarin")

TONGUE = "四是四，十是十，十四是十四，四十是四十。"
gen = []
for tag in ["sft_female", "sft_avg", "base_mandarin_prompt", "base_sichuan_prompt"]:
    d = os.path.join(os.path.expanduser("~/chuan/eval/final"), tag)
    p = os.path.join(d, "00_0.wav")
    if os.path.exists(p):
        gen.append((tag, p))

# generate the tongue twister fresh for each model so the comparison is on identical text
from cosyvoice.cli.cosyvoice import AutoModel as CModel
d = os.path.join(args.out, "tongue")
os.makedirs(d, exist_ok=True)
for tag, model_dir, spk, prompt in [
        ("tts_sft", os.path.expanduser("~/chuan/models/cosyvoice2-sichuan2"), "sichuan_female", None),
        ("tts_base", "/home/whu/models/CosyVoice2-0.5B", None,
         ("希望你以后能够做的比我还好呦。", "/home/whu/CosyVoice/asset/zero_shot_prompt.wav"))]:
    m = CModel(model_dir=model_dir, fp16=True)
    for k in range(3):
        torch.manual_seed(2000 + k)
        np.random.seed(2000 + k)
        it = (m.inference_sft(TONGUE, spk, stream=False) if spk
              else m.inference_zero_shot(TONGUE, prompt[0], prompt[1], stream=False))
        wav = np.concatenate([o["tts_speech"].squeeze().cpu().numpy() for o in it])
        p = os.path.join(d, "%s_%d.wav" % (tag, k))
        sf.write(p, wav, m.sample_rate)
        print("gen", p, round(len(wav) / m.sample_rate, 2), flush=True)
    del m
    torch.cuda.empty_cache()

for tag in ["tts_sft", "tts_base"]:
    items = [(p, TONGUE) for p in sorted(glob.glob(os.path.join(d, tag + "_*.wav")))]
    rep[tag] = collect(items, tag)

json.dump(rep, open(os.path.join(args.out, "minshi.json"), "w"), indent=1)
print("DONE_MINSHI", flush=True)
