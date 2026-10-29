"""Manifest v2: clean Sichuan transcripts only + KeSpeech Southwestern + Mandarin replay.

Drops the 730 ASR-auto-labelled aotemande clips (their labels contain ASR errors such
as 医万/真溪, which teach the TTS wrong text-to-speech alignments) and replaces them
with KeSpeech Southwestern Mandarin, which has human transcripts. The 400 KeSpeech
utterances used by the ASR evaluation are excluded so that set stays held out.
"""
import os, json, random, glob, soundfile as sf, pyarrow.parquet as pq

ROOT = os.path.expanduser("~/chuan/data")
SFT = os.path.join(ROOT, "sft")
MAN = os.path.join(ROOT, "raw", "mandarin")
KS = os.path.join(ROOT, "raw", "kespeech")
KSW = os.path.join(ROOT, "raw", "kespeech_sw_train")


def dur(p):
    try:
        return sf.info(p).duration
    except Exception:
        return 0.0


rows = []
# 1) Sichuan transcribed (magic_tts + xiNan), human text
for r in json.load(open(os.path.join(SFT, "all_sichuan.json"))):
    if len(r) == 5:
        rows.append(list(r))
n_sichuan = len(rows)
print("clean sichuan", n_sichuan)

# 2) KeSpeech Southwestern Mandarin, human text, eval 400 held out
eval_ids = {os.path.basename(r["wav"])[:-4]
            for r in json.load(open(os.path.join(ROOT, "eval", "kespeech_sw.json")))}
os.makedirs(KSW, exist_ok=True)
ks = []
for f in sorted(glob.glob(os.path.join(KS, "*.parquet"))):
    for r in pq.read_table(f).to_pylist():
        if r.get("Dialect") != "Southwestern":
            continue
        txt = (r.get("Text") or "").strip()
        uid = r.get("ID") or ""
        if not txt or not uid or uid in eval_ids:
            continue
        p = os.path.join(KSW, uid + ".wav")
        if not os.path.exists(p):
            open(p, "wb").write(r["audio"]["bytes"])
        d = dur(p)
        if 1.0 <= d <= 25 and len(txt) >= 2:
            ks.append([uid, p, txt, "ks_" + uid.split("_")[0], d])
print("kespeech southwestern added", len(ks))
rows.extend(ks)

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
        man.append([fn[:-4], p, r["sentence"], "thchs_" + fn.split("_")[0], d])
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
json.dump(rows, open(os.path.join(SFT, "all_rows_v2.json"), "w"), ensure_ascii=False)
print("TOTAL", len(rows), "hours", round(sum(r[4] for r in rows) / 3600, 2))
