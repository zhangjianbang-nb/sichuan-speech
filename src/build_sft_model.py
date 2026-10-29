"""Build an inference-ready CosyVoice2 model dir from the SFT llm checkpoint."""
import os, sys, shutil, torch, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prompt_wav", required=True)
ap.add_argument("--prompt_text", required=True)
ap.add_argument("--spk_id", default="sichuan_female")
ap.add_argument("--extra_spk", nargs="*", default=[])
args = ap.parse_args()

PRE = "/home/whu/models/CosyVoice2-0.5B"
os.makedirs(args.out, exist_ok=True)

# link everything except llm.pt / spk2info.pt
for name in os.listdir(PRE):
    if name in ("llm.pt", "spk2info.pt"):
        continue
    src = os.path.join(PRE, name)
    dst = os.path.join(args.out, name)
    if os.path.exists(dst):
        continue
    if os.path.isdir(src):
        os.symlink(src, dst)
    else:
        os.symlink(src, dst)

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
for spec in args.extra_spk:
    wav, text, sid = spec.split("|", 2)
    m.add_zero_shot_spk(text, wav, sid)
m.save_spkinfo()
print("spks:", m.list_available_spks())
print("DONE_BUILD")
