"""Cross-corpus sanity check: does the dialect classifier generalize beyond the training corpora?"""
import os, sys, json, random
import numpy as np
import torch, soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialect_clf_lib import Net, mel, SR

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
clf = Net().cuda()
clf.load_state_dict(torch.load(os.path.expanduser("~/chuan/exp/dialect_clf/clf.pt"), map_location="cuda"))
clf.eval()


@torch.no_grad()
def prob(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    n = int(3.0 * SR)
    if len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    ps = []
    for s in range(0, max(1, len(wav) - n + 1), n):
        seg = wav[s:s + n]
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        ps.append(torch.softmax(clf(mel(seg).unsqueeze(0).cuda()), 1)[0, 1].item())
    return float(np.mean(ps))


for name in ["kespeech_sw", "kespeech_mandarin"]:
    rows = json.load(open(os.path.expanduser("~/chuan/data/eval/%s.json" % name)))
    ps = [prob(r["wav"]) for r in rows]
    acc = np.mean([p > 0.5 for p in ps])
    print("%s n=%d mean_p_sichuan=%.3f  frac>0.5=%.3f" % (name, len(ps), np.mean(ps), acc), flush=True)
print("DONE_CROSSCHECK")
