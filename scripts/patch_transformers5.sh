#!/bin/bash
# Patch CosyVoice's cached-decoding attention mask so it works on transformers>=5.
#
# On transformers 5.x, forward_one_step passes a length-1 attention_mask while the KV
# cache already holds the full prompt, so every past position gets masked out and the
# model can only see the SOS token -> gibberish speech. See docs/incident-transformers5.md.
#
# Usage: bash scripts/patch_transformers5.sh [COSYVOICE_ROOT]
#        (default: $SICHUAN_COSYVOICE_HOME, else ~/CosyVoice)
set -e
ROOT="${1:-${SICHUAN_COSYVOICE_HOME:-$HOME/CosyVoice}}"
F="$ROOT/cosyvoice/llm/llm.py"
[ -f "$F" ] || { echo "not found: $F"; exit 1; }
[ -f "$F.bak-chuan" ] || cp "$F" "$F.bak-chuan"

python3 - "$F" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if "Rebuild a full-length mask" in s:
    print("already patched")
    sys.exit(0)
old = """    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model("""
new = """    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        # Rebuild a full-length mask: the caller passes a 1-token mask during cached
        # decoding, and transformers>=5 expands it literally, masking out every past key.
        past = 0
        if cache is not None:
            if hasattr(cache, "get_seq_length"):
                past = cache.get_seq_length()
            elif isinstance(cache, (tuple, list)) and len(cache) and cache[0] is not None:
                past = cache[0][0].shape[-2]
        total = past + xs.shape[1]
        if input_masks.shape[-1] != total:
            input_masks = torch.ones((xs.shape[0], total), dtype=torch.bool, device=xs.device)
        outs = self.model("""
if old not in s:
    print("anchor not found; the file layout changed, patch by hand")
    sys.exit(2)
open(p, "w").write(s.replace(old, new))
print("patched", p)
PY
