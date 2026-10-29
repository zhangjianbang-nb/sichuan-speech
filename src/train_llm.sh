#!/bin/bash
# CosyVoice2-0.5B LLM domain-transfer SFT on Sichuan dialect data (GPU1 only)
set -x
cd /home/whu/CosyVoice/examples/libritts/cosyvoice2 || exit 1
. ./path.sh || exit 1
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/whu/CosyVoice:/home/whu/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH

/home/whu/audio/venv/bin/torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1986 --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
  ../../../cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config /home/whu/chuan/conf/sft.yaml \
  --train_data /home/whu/chuan/data/sft/train/parquet/data.list \
  --cv_data /home/whu/chuan/data/sft/dev/parquet/data.list \
  --qwen_pretrain_path /home/whu/models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --onnx_path /home/whu/models/CosyVoice2-0.5B \
  --model llm \
  --checkpoint /home/whu/models/CosyVoice2-0.5B/llm.pt \
  --model_dir /home/whu/chuan/exp/llm \
  --tensorboard_dir /home/whu/chuan/tb/llm \
  --ddp.dist_backend nccl \
  --num_workers 2 \
  --prefetch 100 \
  --pin_memory \
