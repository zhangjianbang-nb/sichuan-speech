import re, os, shutil

src = "/home/whu/CosyVoice/examples/libritts/cosyvoice2/conf/cosyvoice2.yaml"
dst_dir = os.path.expanduser("~/chuan/conf")
os.makedirs(dst_dir, exist_ok=True)
dst = os.path.join(dst_dir, "sft.yaml")
s = open(src).read()

# SFT: condition on speaker embedding (learn dialect voice from x-vector)
s = s.replace("use_spk_embedding: False # change to True during sft", "use_spk_embedding: True")
# shorter epochs / denser logging
s = re.sub(r"max_epoch: 200", "max_epoch: 30", s)
s = re.sub(r"log_interval: 100", "log_interval: 20", s)
# keep lr 1e-5 constantlr as recipe suggests for sft
open(dst, "w").write(s)
print("wrote", dst)
for k in ["use_spk_embedding", "max_epoch", "log_interval", "lr:", "scheduler:"]:
    for line in s.splitlines():
        if k in line:
            print(line.strip())
