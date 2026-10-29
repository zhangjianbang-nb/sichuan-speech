import os, json, random

ROOT = os.path.expanduser("~/chuan/data/sft")
rows = json.load(open(os.path.join(ROOT, "all_sichuan.json")))
rows = [r for r in rows if len(r) == 5]
random.seed(1986)
random.shuffle(rows)
n_dev = max(20, int(len(rows) * 0.03))
dev, train = rows[:n_dev], rows[n_dev:]

for name, rs in [("train", train), ("dev", dev)]:
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    with open(d + "/wav.scp", "w") as fw, open(d + "/text", "w") as ft, open(d + "/utt2spk", "w") as fu:
        for utt, path, text, spk, dur in rs:
            utt = utt.replace(" ", "_")
            fw.write("%s %s\n" % (utt, path))
            ft.write("%s %s\n" % (utt, text))
            fu.write("%s %s\n" % (utt, spk))
    spk2utt = {}
    for utt, path, text, spk, dur in rs:
        spk2utt.setdefault(spk, []).append(utt.replace(" ", "_"))
    with open(d + "/spk2utt", "w") as f:
        for s, u in spk2utt.items():
            f.write("%s %s\n" % (s, " ".join(u)))
    print(name, len(rs), "spks", len(spk2utt))
