"""Build an inference-ready CosyVoice2 model dir from an SFT llm checkpoint.

Adds two Sichuan speakers to spk2info:
  sichuan_female  - the MagicHub Chengdu female prompt (natural, studio-clean)
  sichuan_avg     - the mean campplus embedding of the KeSpeech Southwestern speakers
                    (a "typical Sichuan voice"; the LLM controls accent, the flow uses
                    this embedding only for timbre)
"""
import os, sys, argparse, torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prompt_wav", required=True)
ap.add_argument("--prompt_text", required=True)
ap.add_argument("--spk_id", default="sichuan_female")
ap.add_argument("--emb_pt", default=os.path.expanduser("~/chuan/data/sft/train/utt2embedding.pt"))
ap.add_argument("--spk_prefix", default="ks_", help="only average utts whose speaker starts with this")
ap.add_argument("--utt2spk", default=os.path.expanduser("~/chuan/data/sft/train/utt2spk"))
ap.add_argument("--avg_spk_id", default="sichuan_avg")
args = ap.parse_args()

PRE = "/home/whu/models/CosyVoice2-0.5B"
os.makedirs(args.out, exist_ok=True)

for name in os.listdir(PRE):
    if name in ("llm.pt", "spk2info.pt"):
        continue
    dst = os.path.join(args.out, name)
    if not os.path.exists(dst):
        os.symlink(os.path.join(PRE, name), dst)

sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
sd.pop("epoch", None)
sd.pop("step", None)
torch.save(sd, os.path.join(args.out, "llm.pt"))
print("llm.pt written", len(sd), "keys")

sys.path.append("/home/whu/CosyVoice")
sys.path.append("/home/whu/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel

m = AutoModel(model_dir=args.out, fp16=True)
m.add_zero_shot_spk(args.prompt_text, args.prompt_wav, args.spk_id)

for sid, v in m.frontend.spk2info.items():
    if isinstance(v, dict) and "embedding" not in v and "llm_embedding" in v:
        v["embedding"] = v["llm_embedding"]

if os.path.exists(args.emb_pt):
    d = torch.load(args.emb_pt, map_location="cpu", weights_only=False)
    keep = set()
    if os.path.exists(args.utt2spk):
        for line in open(args.utt2spk):
            u, spk = line.split()[:2]
            if spk.startswith(args.spk_prefix):
                keep.add(u)
    vecs = [(v if torch.is_tensor(v) else torch.tensor(v)).reshape(1, -1)
            for k, v in d.items() if (not keep or k in keep)]
    if vecs:
        emb = torch.cat(vecs, 0).mean(0, keepdim=True)
        emb = torch.nn.functional.normalize(emb, dim=1)
        print("avg embedding from", len(vecs), "utts (spk prefix %r)" % args.spk_prefix, tuple(emb.shape))
        m.frontend.spk2info[args.avg_spk_id] = {"embedding": emb}
    else:
        print("no utts matched speaker prefix", args.spk_prefix)

m.save_spkinfo()
for sid, v in m.frontend.spk2info.items():
    print("spk", sid, list(v.keys()))
print("DONE_BUILD")
