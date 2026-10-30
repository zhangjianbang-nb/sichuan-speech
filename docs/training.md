# 训练全流程

两段训练：CosyVoice2-0.5B 的 LLM 做 TTS 方言迁移，SenseVoiceSmall 做 ASR 方言迁移。
全部在单张 RTX 4090（GPU1，24 GB）上完成，fp32，不动 torch、不装新算子。GPU0 留给 ComfyUI。

数据从哪来、怎么清洗，见 [data-collection.md](data-collection.md)。踩过的坑与经验，
见 [training-notes.md](training-notes.md)。

## 0. 环境

```bash
# 远端机器（示例：192.168.3.178）
cd ~/chuan/repo
python -m venv ~/audio/venv          # 已有可跳过
~/audio/venv/bin/pip install -e ".[train]"

export SICHUAN_COSYVOICE_HOME=~/CosyVoice
export SICHUAN_BASE_DIR=~/models/CosyVoice2-0.5B
export SICHUAN_DATA_ROOT=~/chuan/data
export SICHUAN_WORK_ROOT=~/chuan
export SICHUAN_GPU=1
```

`[train]` 额外依赖：torch/torchaudio（按机器装，不要用 pip 重装）、funasr、librosa、
pyarrow、scikit-learn、onnxruntime。CosyVoice 本身不打包进 wheel，用 `SICHUAN_COSYVOICE_HOME`
指过去。

### 两处必须先打的源码补丁

1. **transformers ≥ 5 的 KV cache mask**（不打会出现乱码语音）：

   ```bash
   bash scripts/patch_transformers5.sh
   ```

   原理与现场取证见 [incident-transformers5.md](incident-transformers5.md)。

2. **bf16 权重转 fp32**：在 `cosyvoice/bin/train.py` 载入 checkpoint 之后加一行
   `model = model.float()`，并且**不要**传 `--use_amp`。原因见
   [training-notes.md](training-notes.md) §1。

## 1. 数据准备

```bash
python scripts/extract_kespeech.py        # 400 条西南 + 300 条普通话评测集
python scripts/prep_sichuan.py            # 1853 条人工文本川话清单
python scripts/prep_manifest.py           # 5046 条 / 9.12 h，train 4895 + dev 151
```

## 2. TTS 特征抽取（约 13 分钟）

CosyVoice2 用 Kaldi 风格目录 + 预抽特征：

```
data/sft/{train,dev}/
  wav.scp text utt2spk spk2utt          # 文本清单（上一步生成）
  spk2embedding.pt utt2embedding.pt     # campplus.onnx 抽的说话人向量
  utt2speech_token.pt                   # speech_tokenizer_v2.onnx 抽的离散 token
  parquet/{data.list,*.tar}             # make_parquet_list.py 打包
```

```bash
python scripts/prep_features.py --splits train dev --num-processes 10 --utts-per-parquet 1000
```

三个工具在 `<cosyvoice_home>/tools/`（不在 `examples/` 下），必须用装了 torch 的解释器跑；
`prepare_features()` 会自动设好 `PYTHONPATH`（含 `third_party/Matcha-TTS`）。

## 3. TTS 训练配置

```bash
python scripts/make_conf.py --lr 3e-6 --max-epoch 20 --log-interval 20
# 写出 $SICHUAN_WORK_ROOT/conf/sft.yaml；仓库里的 conf/sft2.yaml 就是这次实际用的配置
```

相对官方 `cosyvoice2.yaml` 只改这几处：

| 项 | 值 | 理由 |
|---|---|---|
| `padding.use_spk_embedding` | `True` | 多说话人语料，必须按说话人条件化 |
| `train_conf.optim_conf.lr` | `3e-6`（首轮用过 `1e-5`） | 1e-5 在 6 小时数据上一轮就过拟合 |
| `train_conf.scheduler` | `constantlr` | 官方 SFT 推荐 |
| `train_conf.max_epoch` | `20`（实际跑到 8 轮就停） | 每轮存 checkpoint，按 CV loss 选 |
| `accum_grad` | `2` | 单卡等效 batch 放大 |
| `log_interval` | `20` | 便于观察 |

## 4. 启动训练

```bash
bash scripts/train_llm.sh
```

脚本用 `torchrun --nproc_per_node=1` 拉起官方 `train.py --model llm`，路径全部由环境变量给出
（`SICHUAN_TORCHRUN` 可指到 venv 里的绝对路径）。每轮约 2.5 分钟（454 步），显存约 16 GB。

### 第二轮 CV 曲线（lr 3e-6，9.12 小时）

| epoch | CV loss | token acc |
|---|---|---|
| 0 | 3.5478 | 0.2191 |
| 1 | 3.4975 | 0.2239 |
| 2 | 3.4816 | 0.2246 |
| 3 | 3.4751 | 0.2258 |
| **4（选用）** | **3.4716** | **0.2275** |
| 5 | 3.4878 | 0.2263 |
| 6 | 3.5052 | 0.2263 |
| 7 | 3.5428 | 0.2256 |
| 8 | 3.5921 | 0.2209 |

epoch 4 触底，之后单调回升 → 停止训练。选它是因为 CV loss 最低，**不是**因为 CER 最好
（20 句批量上 epoch 2 的 CER 更漂亮，见 [evaluation.md](evaluation.md) §2.2）。

## 5. 组装推理模型

```bash
python scripts/build_sft_model.py \
  --ckpt "$SICHUAN_WORK_ROOT/exp/llm2/epoch_4_whole.pt" \
  --out  "$SICHUAN_WORK_ROOT/models/cosyvoice2-sichuan"
```

这一步做四件事：

1. 把 base 模型里没改的 flow / HiFi-GAN / tokenizer 文件软链过去，只替换 `llm.pt`；
2. 注册微调说话人 `sichuan_female`（用 MagicHub 成都女声的 campplus 向量）；
3. 用 KeSpeech 训练集全部说话人向量的均值注册 `sichuan_avg`；
4. **把 `llm_embedding` 复制一份到 `embedding`**：`add_zero_shot_spk()` 写的是前者，
   `frontend_sft()` 读的是后者，不复制会 `KeyError: 'embedding'`（详见 training-notes §7）。

## 6. ASR 训练

```bash
python scripts/train_asr.py --profile round2        # 4 epochs, lr 4e-6, batch 3, 10 s
```

标签是「特殊 token 前缀 + 正文」：

```python
ids = [24884, 24993, 25004, 25017] + tokenizer.encode(text)
#      <|zh|> <|Speech|> <|NEUTRAL|> <|woitn|> + 正文
```

`24884` 起是特殊 token 区，CTC 词表 25055，`blank_id=0`，`ignore_id=-1`。
情感位必须是 `25004`（`<|NEUTRAL|>`）；第一轮误填 `0` 让情感头学到了 blank。

第二轮 DEV_CER（域内 dev 151 条）：0.1277 → 0.1159 → 0.1134 → **0.1131**。
域内 dev 只用来选 epoch，正式结论看跨说话人留出集（`docs/evaluation.md` §1）。

显存：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，batch 3 / 10 秒稳定 5.8 GB；
batch 4 / 20 秒会 OOM。

## 7. 评测

```bash
python scripts/eval_asr.py --ckpt "$SICHUAN_WORK_ROOT/exp/asr2/epoch_2.pt"   # 双向 CER
python scripts/eval_tts.py                                                    # 锚点 + 四集合
python scripts/sweep_epochs.py --n-sent 20                                    # 逐检查点口音指标
python scripts/eval_accent.py                                                 # 声学 + 分类器
python scripts/check_speaker_sim.py                                           # 音色余弦
python scripts/eval_minshi.py                                                 # 失败的平翘舌检验
```

结果写进 `$SICHUAN_EVAL_ROOT` 下的 JSON，口径与解释见 [evaluation.md](evaluation.md)。

## 8. 说话人向量的两个坑

1. `AutoModel.add_zero_shot_spk()` 存进 `spk2info` 的字典里有 `llm_embedding`/`flow_embedding`，
   但 `frontend_sft()` 读的是 `spk2info[spk_id]['embedding']`。两个函数对 schema 的假设不一致。
2. **CosyVoice2 的 LLM 不用说话人向量**。读 `Qwen2LM.inference` 可以看到 `lm_input` 只由
   `[sos, text_emb, task_id, prompt_speech_token]` 拼成，没有 embedding 项。
   所以发音/口音完全由 LLM 的文本→语音 token 映射决定，说话人向量只影响 flow 的音色。
   这就是为什么换一个说话人向量能换音色（`sichuan_female` vs `sichuan_avg`），但口音不变
   ——两者的回读转写完全一致。

## 9. 时间与资源

| 阶段 | 耗时 | 显存 |
|---|---|---|
| 数据准备 + 评测集抽取 | 5–10 分钟 | CPU |
| 特征抽取（embedding/token/parquet） | 约 13 分钟 | CPU + ONNX |
| TTS LLM SFT 每轮 | 约 2.5 分钟 | 16 GB |
| TTS 共 9 轮（跑到 epoch 8） | 约 23 分钟 | 16 GB |
| ASR 每轮（4895 条） | 约 12 分钟 | 5.8 GB |
| ASR 共 4 轮 | 约 48 分钟 | 5.8 GB |
| 全套评测 | 30–40 分钟 | CPU 为主 |

## 10. 没做的事

1. **flow 与 HiFi-GAN 没有微调**。官方 SFT 配方可以同时训 flow，理论上能改善音色保真和
   韵律。没做是因为无法用客观指标验证收益（flow 的改善只能靠听），时间预算内不冒险。
2. **没做 RL**。方言 TTS 缺少可靠的奖励信号（没有方言判别器、没有可用的发音评分器），
   强行上 RL 只会优化到一个错误的代理目标。ASR 侧有 CER 这种天然奖励，但监督微调已经收敛，
   RL 的边际收益不足以支撑风险。
