"""Compare base CosyVoice2 vs Sichuan-SFT on the same sentences: dialect-ness + ASR CER."""
import os, sys, json, glob, argparse
import numpy as np
import soundfile as sf
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--sft_dir", default=os.path.expanduser("~/chuan/models/cosyvoice2-sichuan"))
ap.add_argument("--sft_spk", default="sichuan_female")
ap.add_argument("--out", default=os.path.expanduser("~/chuan/eval/round"))
ap.add_argument("--prompt_wav", default="")
ap.add_argument("--prompt_text", default="")
ap.add_argument("--tag", default="sft")
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
        if mode == "zero_shot":
            gen_iter = model.inference_zero_shot(t, prompt[1], prompt[0], stream=False)
        else:
            gen_iter = model.inference_sft(t, spk, stream=False)
        for j, o in enumerate(gen_iter):
            wav = o["tts_speech"].squeeze().cpu().numpy()
            p = os.path.join(outdir, "%02d_%d.wav" % (i, j))
            sf.write(p, wav, model.sample_rate)
            outs.append(p)
        paths.append(outs)
    return paths


# ---- dialect classifier ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialect_clf_lib import Net, mel, SR, N_MELS

clf = Net().cuda()
clf.load_state_dict(torch.load(os.path.expanduser("~/chuan/exp/dialect_clf/clf.pt"), map_location="cuda"))
clf.eval()


@torch.no_grad()
def dialect_prob(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    n = int(3.0 * SR)
    if len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    # average over windows
    probs = []
    for s in range(0, max(1, len(wav) - n + 1), n):
        seg = wav[s:s + n]
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        m = mel(seg).unsqueeze(0)
        probs.append(torch.softmax(clf(m.cuda()), 1)[0, 1].item())
    return float(np.mean(probs))


# ---- ASR (base SenseVoice on CPU) ----
from funasr import AutoModel as FModel
asr = FModel(model="/home/whu/models/SenseVoiceSmall", device="cpu", disable_update=True)


def asr_text(path):
    import re
    r = asr.generate(input=path, cache={}, language="zh", use_itn=True, batch_size_s=60)
    t = re.sub(r"<\|[^|]*\|>", "", r[0]["text"]).strip()
    return t


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


from cosyvoice.cli.cosyvoice import AutoModel as CModel
prompt = pick_prompt()
print("prompt:", prompt[0], prompt[2], prompt[1], flush=True)

report = {}
base = CModel(model_dir="/home/whu/models/CosyVoice2-0.5B", fp16=True)
p_base = gen(base, SICHUAN, "zero_shot", os.path.join(args.out, "base_sichuan"), prompt=prompt)
del base
torch.cuda.empty_cache()

sft = CModel(model_dir=args.sft_dir, fp16=True)
p_sft = gen(sft, SICHUAN, "sft", os.path.join(args.out, "sft_sichuan"), spk=args.sft_spk)
p_sft_man = gen(sft, MANDARIN, "sft", os.path.join(args.out, "sft_mandarin"), spk=args.sft_spk)
del sft
torch.cuda.empty_cache()


def score(paths, refs, name):
    rows = []
    tot_e = tot_n = 0
    for i, outs in enumerate(paths):
        for p in outs:
            dp = dialect_prob(p)
            hyp = asr_text(p)
            e, n = cer(refs[i], hyp)
            tot_e += e
            tot_n += n
            rows.append({"wav": p, "dialect_prob": round(dp, 4), "hyp": hyp,
                         "cer": round(e / max(1, n), 4)})
    report[name] = {
        "mean_dialect_prob": round(float(np.mean([r["dialect_prob"] for r in rows])), 4),
        "mean_cer": round(tot_e / max(1, tot_n), 4),
        "rows": rows,
    }
    print(name, "dialect_prob", report[name]["mean_dialect_prob"],
          "cer", report[name]["mean_cer"], flush=True)


score(p_base, SICHUAN, "base_zero_shot_sichuan_text")
score(p_sft, SICHUAN, "sft_sichuan_text")
score(p_sft_man, MANDARIN, "sft_mandarin_text")

json.dump(report, open(os.path.join(args.out, "report_%s.json" % args.tag), "w"),
          ensure_ascii=False, indent=1)
print("DONE_EVAL", flush=True)
