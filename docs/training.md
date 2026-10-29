# 训练配方

两段训练：CosyVoice2 的 LLM 做 TTS 方言迁移，SenseVoiceSmall 做 ASR 方言迁移。
全部在单张 RTX 4090（GPU1）上完成，fp32，不装/不改 torch。GPU0 留给 ComfyUI。

## 一、TTS：CosyVoice2-0.5B LLM SFT

### 数据流水线

CosyVoice2 用的是 Kaldi 风格目录 + 预抽取特征：

```
data/sft/{train,dev}/
  wav.scp text utt2spk spk2utt          # 文本清单
  spk2embedding.pt utt2embedding.pt     # campplus.onnx 抽的说话人向量
  utt2speech_token.pt                   # speech_tokenizer_v2.onnx 抽的离散 token
  parquet/{data.list,*.tar}             # make_parquet_list.py 打包
```

命令（注意 `tools/` 在仓库根，不在 example 目录下，且必须用 venv 的 python）：

```bash
cd /home/whu/CosyVoice
export PYTHONPATH=/home/whu/CosyVoice:/home/whu/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH
PY=/home/whu/audio/venv/bin/python
for x in train dev; do
  D=/home/whu/chuan/data/sft/$x
  $PY tools/extract_embedding.py     --dir $D --onnx_path .../campplus.onnx
  $PY tools/extract_speech_token.py  --dir $D --onnx_path .../speech_tokenizer_v2.onnx
  rm -rf $D/parquet; mkdir -p $D/parquet
  $PY tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 10 \
      --src_dir $D --des_dir $D/parquet
done
```

5046 条 / 9.12 小时（train 4895 + dev 151），三步合计约 13 分钟。

### 训练配置（相对官方 `cosyvoice2.yaml` 的改动）

| 项 | 值 | 理由 |
|---|---|---|
| `padding.use_spk_embedding` | `True` | 多说话人语料，必须按说话人条件化 |
| `train_conf.optim_conf.lr` | `3e-6`（首轮用过 `1e-5`） | 1e-5 在 6 小时数据上一轮就过拟合，见下 |
| `train_conf.scheduler` | `constantlr` | 官方 SFT 推荐 |
| `train_conf.max_epoch` | `20`（实际跑到 8 轮就停） | 每轮存 checkpoint，按 CV loss 选 |
| `accum_grad` | `2` | 单卡等效 batch 放大 |
| `log_interval` | `20` | 便于观察 |

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

epoch 4 触底，之后单调回升 → 停止训练。每轮 2.5 分钟（454 步），显存约 16 GB。

### 踩过的坑

1. **bf16 预训练权重 + `--use_amp` 直接崩**：
   `RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`。
   修法是在 `train.py` 载入 checkpoint 之后加一句 `model = model.float()`，然后**去掉 `--use_amp`**。
   CosyVoice2-0.5B 的 `CosyVoice-BlankEN/config.json` 里 `torch_dtype` 是 `bfloat16`，
   不显式转 fp32 的话 GradScaler 会撞上不支持 bf16 的 unscale 算子。
2. **过拟合极快**：第一轮 6.08 小时数据、lr 1e-5，CV loss 从 epoch 0 的 3.1697 只降到
   epoch 1 的 3.1344，之后一路涨到 epoch 7 的 4.5628——有效学习只有一轮。
   第二轮把 lr 降到 3e-6，并把语料扩到 9.12 小时、剔除噪声标签，才拿到 9 轮可用曲线。
3. **`torchrun` 不在 PATH**：venv 里没有 `pip` 可执行文件（只有 `python -m pip`），
   `torchrun` 要用绝对路径 `/home/whu/audio/venv/bin/torchrun`。
4. **heredoc 里的引号会被外层 ssh 吃掉**：远程脚本一律本地写好再 `scp`，不要用
   `ssh 'python3 - <<EOF ... EOF'` 里嵌单引号。
5. **`pkill -f "train_asr.py"` 会杀掉自己的 ssh 会话**（exit 255），因为 ssh 命令行里
   含有同样的字符串。用精确 pid 或 `pkill -f` 加更长的区分串。

## 二、ASR：SenseVoiceSmall 全参微调

### 数据格式

SenseVoice 的输入是 fbank+LFR+CMVN（80 mel，lfr_m=7，lfr_n=6，即每帧 560 维）。
文本标签不是纯文字，而是**特殊 token 前缀 + 正文**：

```python
ids = [24884, 24993, 0, 25017] + tokenizer.encode(text)
#      <|zh|> <|Speech|> <blank> <|woitn|> + 正文
```

`24884` 起是特殊 token 区，CTC 词表 25055，blank_id=0，ignore_id=-1。
**情感位必须是 `25004`（`<|NEUTRAL|>`）**，第一轮误填 `0` 会让情感头学到 blank。

### 第二轮配置

`--epochs 4 --batch 3 --lr 4e-6 --max_sec 10`，AdamW + OneCycleLR，
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

DEV_CER（域内 dev 151 条）：0.1277 → 0.1159 → 0.1134 → **0.1131**。
域内 dev 与 KeSpeech 跨说话人留出集不是一回事，正式数字看 `docs/evaluation.md` §1。

**显存坑**：TTS 训练同时占着 GPU 时，batch 4 / 20 秒序列要 5.77 GB 会 OOM，
降到 batch 3 / 12 秒仍要 5.79 GB；最后用 batch 1 / 8 秒，稳定在 5.3 GB。
`expandable_segments:True` 能显著减少碎片。第二轮改成 batch 3 / 10 秒，稳定不 OOM。

### 为什么加普通话回放

只喂方言会让模型把普通话也读成方言（灾难性遗忘）。训练集里混了 900 条 THCHS-30 普通话，
占比 18%。结果是普通话 CER 从 0.0384 降到 0.0340（见评测文档），不退化反降。

### 学习率的教训

第一轮用 lr 2e-5 训 8 轮，域内 DEV_CER 最低 0.0773（epoch 4），但跨说话人留出集上
**比基线更差**（川话 0.1258 vs 0.1078，普通话 0.0449 vs 0.0384）。域内 dev 是同一批
说话人、同一个录音环境，过拟合被掩盖了。第二轮把 lr 降到 4e-6、数据换成干净的
human-transcript 版本，才在两个方向都变好。

**教训：域内 dev 只能用来选 epoch，不能用来下结论。结论必须来自跨说话人留出集。**

## 三、说话人向量的两个坑

1. `AutoModel.add_zero_shot_spk()` 存进 `spk2info` 的字典里有 `llm_embedding`/`flow_embedding`，
   但 `frontend_sft()` 读的是 `spk2info[spk_id]['embedding']` —— 两个函数对 schema 的假设不一致。
   直接 `inference_sft` 会 `KeyError: 'embedding'`。修法：把 `llm_embedding` 复制一份到 `embedding`。
2. **CosyVoice2 的 LLM 不用说话人向量**。读 `Qwen2LM.inference` 可以看到 `lm_input` 只由
   `[sos, text_emb, task_id, prompt_speech_token]` 拼成，没有 embedding 项。
   所以：**发音/口音完全由 LLM 的文本→语音 token 映射决定，说话人向量只影响 flow 的音色**。
   这也是为什么 SFT 模型换一个说话人向量就能换音色（`sichuan` vs `sichuan_avg`），但口音不变
   ——两者的回读转写完全一致，就是这个原因。

## 四、没做的事

1. **flow 与 HiFi-GAN 没有微调**。官方 SFT 配方可以同时训 flow，理论上能改善音色保真和
   韵律。本仓库没做，因为无法用客观指标验证收益（flow 的改善只能靠听），
   在时间预算内不冒险。
2. **没做 RL**。用户提到"必要时 RL 训练"，但方言 TTS 缺少可靠的奖励信号（没有方言
   判别器、没有可用的发音评分器），强行上 RL 只会优化到一个错误的代理目标。
   ASR 侧倒是有 CER 这种天然奖励，但监督微调已经收敛，RL 的边际收益不足以支撑风险。
