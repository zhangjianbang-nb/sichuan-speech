#!/bin/bash
set -e
cd /home/whu/CosyVoice
export PYTHONPATH=/home/whu/CosyVoice:/home/whu/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH
PY=/home/whu/audio/venv/bin/python
for x in train dev; do
  D=/home/whu/chuan/data/sft/$x
  echo "== $x embedding $(date +%H:%M:%S)"
  $PY tools/extract_embedding.py --dir $D --onnx_path /home/whu/models/CosyVoice2-0.5B/campplus.onnx 2>&1 | tail -2
  echo "== $x tokens $(date +%H:%M:%S)"
  $PY tools/extract_speech_token.py --dir $D --onnx_path /home/whu/models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx 2>&1 | tail -2
  rm -rf $D/parquet; mkdir -p $D/parquet
  echo "== $x parquet $(date +%H:%M:%S)"
  $PY tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 10 --src_dir $D --des_dir $D/parquet 2>&1 | tail -2
done
wc -l /home/whu/chuan/data/sft/train/parquet/data.list /home/whu/chuan/data/sft/dev/parquet/data.list
echo PREP_DONE $(date +%H:%M:%S)
