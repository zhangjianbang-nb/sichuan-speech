#!/bin/bash
# Launch the CosyVoice2 LLM SFT run with torchrun.
#
# Everything is environment-driven; the defaults reproduce the delivered run:
#   SICHUAN_COSYVOICE_HOME  ~/CosyVoice
#   SICHUAN_DATA_ROOT       ~/chuan/data
#   SICHUAN_WORK_ROOT       ~/chuan
#   SICHUAN_BASE_DIR        ~/models/CosyVoice2-0.5B
#   SICHUAN_SFT_CONF        $SICHUAN_WORK_ROOT/conf/sft.yaml
#   SICHUAN_EXP_DIR         $SICHUAN_WORK_ROOT/exp/llm2
#   SICHUAN_TB_DIR          $SICHUAN_WORK_ROOT/tb/llm2
#   SICHUAN_TORCHRUN        torchrun
#   SICHUAN_GPU             1
#   SICHUAN_RDZV_PORT       1235
#
# Before the first run, apply the two source edits documented in docs/training-notes.md:
#   (1) bash scripts/patch_transformers5.sh   -- KV-cache mask fix for transformers>=5
#   (2) add `model = model.float()` after the checkpoint load in cosyvoice/bin/train.py
#       and do NOT pass --use_amp: bf16 weights crash GradScaler.
set -euo pipefail

CV=${SICHUAN_COSYVOICE_HOME:-$HOME/CosyVoice}
WORK=${SICHUAN_WORK_ROOT:-$HOME/chuan}
DATA=${SICHUAN_DATA_ROOT:-$WORK/data}
BASE=${SICHUAN_BASE_DIR:-$HOME/models/CosyVoice2-0.5B}
CONF=${SICHUAN_SFT_CONF:-$WORK/conf/sft.yaml}
EXP=${SICHUAN_EXP_DIR:-$WORK/exp/llm2}
TB=${SICHUAN_TB_DIR:-$WORK/tb/llm2}
TORCHRUN=${SICHUAN_TORCHRUN:-torchrun}
PORT=${SICHUAN_RDZV_PORT:-1235}

export CUDA_VISIBLE_DEVICES=${SICHUAN_GPU:-1}
export PYTHONPATH="$CV:$CV/third_party/Matcha-TTS:${PYTHONPATH:-}"

cd "$CV/examples/libritts/cosyvoice2"
# shellcheck disable=SC1091
. ./path.sh

exec "$TORCHRUN" --nnodes=1 --nproc_per_node=1 --rdzv_id=1987 --rdzv_backend=c10d \
  --rdzv_endpoint="localhost:$PORT" \
  "$CV/cosyvoice/bin/train.py" --train_engine torch_ddp --config "$CONF" \
  --train_data "$DATA/sft/train/parquet/data.list" \
  --cv_data "$DATA/sft/dev/parquet/data.list" \
  --qwen_pretrain_path "$BASE/CosyVoice-BlankEN" --onnx_path "$BASE" \
  --model llm --checkpoint "$BASE/llm.pt" --model_dir "$EXP" \
  --tensorboard_dir "$TB" --ddp.dist_backend nccl \
  --num_workers 2 --prefetch 100 --pin_memory
