"""Fair ASR comparison: the base funasr path must run with use_itn=False, otherwise
"七千多种" is emitted as "7000多种" and its CER is inflated relative to the finetuned
model, which never applies ITN.
"""
import os, sys, json, re
import numpy as np

sys.path.append("/home/whu/chuan/work")
from funasr import AutoModel as F

base = F(model="/home/whu/models/SenseVoiceSmall", device="cpu", disable_update=True)


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
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return int(d[len(r), len(h)]), len(r)


def score(rows, name, **kw):
    te = tn = 0
    for r in rows:
        h = base.generate(input=r["wav"], cache={}, language="zh", batch_size_s=60, **kw)[0]["text"]
        e, n = cer(r["text"], h)
        te += e
        tn += n
    m = round(te / max(1, tn), 4)
    print("%-36s CER %.4f" % (name, m), flush=True)
    return m


sw = json.load(open(os.path.expanduser("~/chuan/data/eval/kespeech_sw.json")))
man = json.load(open(os.path.expanduser("~/chuan/data/eval/kespeech_mandarin.json")))
out = {"base_sw_noitn": score(sw, "base / southwestern / use_itn=False", use_itn=False),
       "base_man_noitn": score(man, "base / mandarin / use_itn=False", use_itn=False)}
json.dump(out, open(os.path.expanduser("~/chuan/eval/asr_base_noitn.json"), "w"), indent=1)
print("DONE_NOITN", flush=True)
