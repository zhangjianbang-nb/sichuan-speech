import os, json, random, glob, soundfile as sf, pyarrow.parquet as pq

ROOT = os.path.expanduser("~/chuan/data")
SFT = os.path.join(ROOT, "sft")
MAN = os.path.join(ROOT, "raw", "mandarin")


def dur(p):
    try:
        return sf.info(p).duration
    except Exception:
        return 0.0


rows = []
# 1) Sichuan transcribed (magic + xiNan)
for r in json.load(open(os.path.join(SFT, "all_sichuan.json"))):
    if len(r) == 5:
        rows.append(list(r))
n_sichuan = len(rows)

# 2) sichuanhua ASR-labelled
labels = json.load(open(os.path.join(ROOT, "sichuanhua_asr.json")))
for b, t in labels.items():
    p = os.path.join(ROOT, "raw", "sichuanhua", b)
    d = dur(p)
    if 1.0 <= d <= 25 and len(t) >= 2:
        rows.append([b[:-4], p, t, "utt_" + b[:-4], d])
n_add = len(rows) - n_sichuan
print("sichuanhua added", n_add)

# 3) Mandarin replay (THCHS-30)
wavdir = os.path.join(MAN, "wav")
os.makedirs(wavdir, exist_ok=True)
t = pq.read_table(os.path.join(MAN, "thchs_train0.parquet"))
man = []
for r in t.to_pylist():
    fn = os.path.basename(r["audio"]["path"] or "")
    if not fn:
        continue
    p = os.path.join(wavdir, fn)
    if not os.path.exists(p):
        open(p, "wb").write(r["audio"]["bytes"])
    d = dur(p)
    if 1.0 <= d <= 25 and len(r["sentence"]) >= 2:
        spk = "thchs_" + fn.split("_")[0]
        man.append([fn[:-4], p, r["sentence"], spk, d])
random.seed(7)
random.shuffle(man)
man = man[:900]
print("mandarin replay", len(man))
rows.extend(man)

random.seed(1986)
random.shuffle(rows)
n_dev = max(20, int(len(rows) * 0.03))
dev, train = rows[:n_dev], rows[n_dev:]
for name, rs in [("train", train), ("dev", dev)]:
    d = os.path.join(SFT, name)
    os.makedirs(d, exist_ok=True)
    with open(d + "/wav.scp", "w") as fw, open(d + "/text", "w") as ft, open(d + "/utt2spk", "w") as fu:
        for utt, path, text, spk, dur_ in rs:
            utt = utt.replace(" ", "_")
            fw.write("%s %s\n" % (utt, path))
            ft.write("%s %s\n" % (utt, text))
            fu.write("%s %s\n" % (utt, spk))
    spk2utt = {}
    for utt, path, text, spk, dur_ in rs:
        spk2utt.setdefault(spk, []).append(utt.replace(" ", "_"))
    with open(d + "/spk2utt", "w") as f:
        for s, u in spk2utt.items():
            f.write("%s %s\n" % (s, " ".join(u)))
    print(name, len(rs), "spks", len(spk2utt), "hours", round(sum(r[4] for r in rs) / 3600, 2))
json.dump(rows, open(os.path.join(SFT, "all_rows.json"), "w"), ensure_ascii=False)
print("TOTAL", len(rows), "hours", round(sum(r[4] for r in rows) / 3600, 2))
