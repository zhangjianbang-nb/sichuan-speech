import os, json, soundfile as sf

SRC = os.path.expanduser("~/chuan/data/raw/kespeech")
OUT = os.path.expanduser("~/chuan/data/eval")
os.makedirs(OUT, exist_ok=True)

import pyarrow.parquet as pq

south, mand = [], []
for f in ["test-0.parquet", "test-1.parquet"]:
    t = pq.read_table(os.path.join(SRC, f))
    for r in t.to_pylist():
        d = r["Dialect"]
        txt = (r["Text"] or "").strip()
        if not txt:
            continue
        wav = r["audio"]["bytes"]
        if d == "Southwestern":
            south.append((r["ID"], wav, txt))
        elif d == "Mandarin":
            mand.append((r["ID"], wav, txt))

print("southwestern", len(south), "mandarin", len(mand))


def dump(items, sub, limit):
    d = os.path.join(OUT, sub)
    os.makedirs(d, exist_ok=True)
    rows = []
    for i, (uid, wav, txt) in enumerate(items[:limit]):
        p = os.path.join(d, uid + ".wav")
        open(p, "wb").write(wav)
        rows.append({"wav": p, "text": txt})
    json.dump(rows, open(os.path.join(OUT, sub + ".json"), "w"), ensure_ascii=False)
    print(sub, len(rows))


dump(south, "kespeech_sw", 400)
dump(mand, "kespeech_mandarin", 300)
