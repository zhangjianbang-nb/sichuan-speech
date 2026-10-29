"""Compare base SenseVoiceSmall vs Sichuan-finetuned checkpoint on real held-out sets.

KeSpeech provides both Southwestern Mandarin and Mandarin recordings from the same
corpus/channel, so the CER gap between the two is a clean dialect signal and the
Mandarin set is the regression guard.
"""
import os, sys, json, re, argparse
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", default="")
ap.add_argument("--out", default=os.path.expanduser("~/chuan/eval/asr_report.json"))
args = ap.parse_args()
sys.path.append("/home/whu/chuan/work")


def cer(ref, hyp):
    r = [c for c in ref if not c.isspace()]
    h = [c for c in hyp if not c.isspace()]
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return int(d[len(r), len(h)]), len(r)


def norm(t):
    t = re.sub(r"<\|[^|]*\|>", "", t)
    return re.sub(r"[\s，。、？！；：,.?!;:\"'“”‘’（）()《》<>·—\-]", "", t)


def score(rows, transcribe, name):
    te = tn = 0
    bad = []
    for r in rows:
        hyp = transcribe(r["wav"])
        e, n = cer(norm(r["text"]), norm(hyp))
        te += e
        tn += n
        if e:
            bad.append({"ref": r["text"], "hyp": hyp, "cer": round(e / max(1, n), 3)})
    m = round(te / max(1, tn), 4)
    print("%-34s CER %.4f  (%d utts)" % (name, m, len(rows)), flush=True)
    return {"mean_cer": m, "n": len(rows), "errors": bad[:25]}


from funasr import AutoModel as F
base = F(model="/home/whu/models/SenseVoiceSmall", device="cpu", disable_update=True)


def base_tr(p):
    r = base.generate(input=p, cache={}, language="zh", use_itn=True, batch_size_s=60)
    return r[0]["text"]


sw = json.load(open(os.path.expanduser("~/chuan/data/eval/kespeech_sw.json")))
man = json.load(open(os.path.expanduser("~/chuan/data/eval/kespeech_mandarin.json")))
report = {"base": {}, "ft": {}}
report["base"]["sw"] = score(sw, base_tr, "base / kespeech_southwestern")
report["base"]["man"] = score(man, base_tr, "base / kespeech_mandarin")

if args.ckpts:
    import train_asr_lib as tal
    for ck in args.ckpts.split(","):
        ft = tal.load_finetuned(ck, device="cpu")
        tag = os.path.basename(ck)
        report["ft"][tag] = {}
        report["ft"][tag]["sw"] = score(sw, ft.transcribe, "%s / southwestern" % tag)
        report["ft"][tag]["man"] = score(man, ft.transcribe, "%s / mandarin" % tag)
        del ft

json.dump(report, open(args.out, "w"), ensure_ascii=False, indent=1)
print("DONE_ASR_EVAL", flush=True)
