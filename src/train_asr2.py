"""SenseVoiceSmall fine-tune, round 2.

Round 1 (lr 2e-5, 8 epochs, 730 auto-labelled clips) looked good on the in-distribution
dev set but was WORSE than the stock model on held-out KeSpeech once both were scored
with use_itn=False (0.1258 vs 0.1078 southwestern, 0.0449 vs 0.0384 Mandarin).
So: lower LR, fewer epochs, clean labels, KeSpeech Southwestern + Mandarin replay.

Also fixes the emotion slot: the official label prefix is
  [<|zh|>, <|Speech|>, <|NEUTRAL|>, <|woitn|>]
where <|NEUTRAL|> = 25004 (see funasr/models/sense_voice/model.py emo_dict). Round 1 used
0 there, which trained the rich-CE head to predict blank for the emotion slot.
"""
import os, json, random, math, time, argparse
import numpy as np
import soundfile as sf
import torch

from funasr import AutoModel
from funasr.frontends.wav_frontend import WavFrontend

SPECIAL_ZH = 24884        # <|zh|>
SPECIAL_SPEECH = 24993    # <|Speech|>
SPECIAL_NEUTRAL = 25004   # <|NEUTRAL|>
SPECIAL_NOITN = 25017     # <|woitn|>
VOCAB_SPECIAL_START = 24884

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=4e-6)
ap.add_argument("--batch", type=int, default=3)
ap.add_argument("--max_sec", type=float, default=10.0)
ap.add_argument("--out", default=os.path.expanduser("~/chuan/exp/asr2"))
ap.add_argument("--gpu", default="1")
ap.add_argument("--log_every", type=int, default=50)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.makedirs(args.out, exist_ok=True)
dev = os.environ.get("DEVICE", "cuda" if not args.smoke else "cpu")

rows = json.load(open(os.path.expanduser("~/chuan/data/sft/all_rows_v2.json")))
rows = [r for r in rows if len(r) == 5 and r[4] <= args.max_sec]
random.seed(1986)
random.shuffle(rows)
dev_rows = sorted(rows[:200], key=lambda r: r[0])
train = rows[200:]

am = AutoModel(model="/home/whu/models/SenseVoiceSmall", device="cpu", disable_update=True)
model = am.model
tok = am.kwargs["tokenizer"]
fe = WavFrontend(**am.kwargs["frontend_conf"])
if dev == "cuda":
    model = model.cuda()
model.train()
print("params %.1fM  train %d dev %d" % (
    sum(p.numel() for p in model.parameters()) / 1e6, len(train), len(dev_rows)), flush=True)


def load_one(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, 16000).numpy()
    return wav


def make_batch(chunk):
    feats, flens, tids, tlens = [], [], [], []
    for utt, path, text, spk, dur in chunk:
        wav = load_one(path)
        w = torch.from_numpy(wav).unsqueeze(0)
        f, fl = fe(w, torch.tensor([wav.shape[0]]))
        feats.append(f[0])
        flens.append(int(fl[0]))
        ids = [SPECIAL_ZH, SPECIAL_SPEECH, SPECIAL_NEUTRAL, SPECIAL_NOITN] + tok.encode(text)
        tids.append(torch.tensor(ids, dtype=torch.long))
        tlens.append(len(ids))
    maxf = max(f.shape[0] for f in feats)
    dim = feats[0].shape[1]
    fb = torch.zeros(len(feats), maxf, dim)
    for k, f in enumerate(feats):
        fb[k, :f.shape[0]] = f
    maxt = max(t.shape[0] for t in tids)
    tb = torch.full((len(tids), maxt), -1, dtype=torch.long)
    for k, t in enumerate(tids):
        tb[k, :t.shape[0]] = t
    return (fb.to(dev), torch.tensor(flens, dtype=torch.long).to(dev),
            tb.to(dev), torch.tensor(tlens, dtype=torch.long).to(dev))


def batch_iter(data, bs, shuffle=True):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield make_batch([data[j] for j in idx[i:i + bs]])


def greedy_decode(logits, lens):
    out = []
    for k in range(logits.shape[0]):
        ids = logits[k, :int(lens[k])].argmax(dim=-1).tolist()
        prev, res = None, []
        for i in ids:
            if i != prev:
                if i != 0 and i < VOCAB_SPECIAL_START:
                    res.append(i)
                prev = i
        out.append(res)
    return out


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


@torch.no_grad()
def evaluate(limit=None):
    model.eval()
    data = dev_rows[:limit] if limit else dev_rows
    tot_e = tot_n = 0
    preds = []
    pos = 0
    for fb, fl, tb, tl in batch_iter(data, 8, shuffle=False):
        enc, enc_lens = model.encode(fb, fl, tb)
        logits = model.ctc.log_softmax(enc)
        hyps = greedy_decode(logits, enc_lens)
        for h in hyps:
            ref = data[pos][2]
            pos += 1
            hyp = tok.decode(h)
            e, n = cer(ref, hyp)
            tot_e += e
            tot_n += n
            preds.append((ref, hyp))
    model.train()
    return tot_e / max(1, tot_n), preds


if args.smoke:
    fb, fl, tb, tl = make_batch(train[:2])
    loss, stats, weight = model(fb, fl, tb, tl)
    print("smoke loss", float(loss), "weight", float(weight), stats)
    c, preds = evaluate(limit=4)
    for r, h in preds:
        print("REF:", r)
        print("HYP:", h)
    print("smoke cer", c)
    raise SystemExit(0)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps_per_epoch = math.ceil(len(train) / args.batch)
total_steps = steps_per_epoch * args.epochs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

step = 0
t0 = time.time()
for ep in range(args.epochs):
    run_loss = 0.0
    nb = 0
    for fb, fl, tb, tl in batch_iter(train, args.batch, shuffle=True):
        loss, stats, weight = model(fb, fl, tb, tl)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        run_loss += float(loss)
        nb += 1
        step += 1
        if nb % args.log_every == 0:
            print("ep %d step %d/%d loss %.4f lr %.2e elapsed %.0fs" % (
                ep, nb, steps_per_epoch, run_loss / nb, sched.get_last_lr()[0],
                time.time() - t0), flush=True)
    c, preds = evaluate()
    print("EPOCH %d done avg_loss %.4f DEV_CER %.4f" % (ep, run_loss / max(1, nb), c), flush=True)
    for r, h in preds[:2]:
        print("  REF:", r)
        print("  HYP:", h)
    torch.save({"model": model.state_dict(), "epoch": ep, "cer": c},
               os.path.join(args.out, "epoch_%d.pt" % ep))
print("DONE_ASR2", flush=True)
