# 四川方言语音模型：让 CosyVoice2 会说川话、让 SenseVoice 听得懂川话

在单张 RTX 4090（24 GB）上，用 9.12 小时开源川话语料，把普通话 TTS 和 ASR 迁移成四川方言，
并保持普通话能力不退化。仓库包含完整数据流水线、训练脚本、评测脚本、踩坑记录和可直接调用的服务。

> 训练与评测全部实机跑过，数字来自 `results/` 下的原始 JSON，可复算。没有估计值。

## 结果速览

**ASR（KeSpeech 跨说话人留出集，`use_itn=False`）**

| 任务 | 基线 | 微调后 | 变化 |
|---|---|---|---|
| 西南官话 CER（400 条） | 0.1078 | **0.0852** | **−21.0% 相对** |
| 普通话 CER（300 条） | 0.0384 | **0.0340** | **−11.5% 相对** |

方言变好，普通话也一起变好——训练集里按 18% 混了普通话回放。

**TTS（同一句川话文本，两个识别器回读）**

| 音频 | 普通话 ASR 回读 | 川话 ASR 回读 | CER |
|---|---|---|---|
| 原版模型（普通话提示音） | 今天天气好**哎哟**我们出去耍**一下**嘛 | 今天天气好哎呦我们出去耍一哈嘛 | 0.250 |
| 原版模型（川话提示音克隆） | 今天天气好**哎哟**我们出去耍**一下**嘛 | 今天天气好哎哟，我们出去耍一哈嘛 | 0.250 |
| **微调模型** | 今天天气好安逸哦我们出去耍一哈嘛 | 今天天气好安逸哦，我们出去耍一哈嘛。 | **0.000** |

文本是「今天天气好安逸哦，我们出去耍一哈嘛。」。原版模型把「安逸」「一哈」读成
「哎哟」「一下」——**读错字**，不是口音差异；微调模型两个识别器都 0 错。
这一句是单句示例；20 句批量上微调的平均 CER 是 0.0483、原版是 0.0552，优势没有单句这么夸张
（见 `docs/evaluation.md` §2.2）。

**其他证据**

| 指标 | 结果 |
|---|---|
| 音色相似度（campplus 余弦） | 0.678（目标说话人 vs 合成音频） |
| 普通话 TTS 回归 | 4 句普通话文本 CER 0.0000 |
| 合成川话口音强度（普通话 ASR CER） | 0.0748 vs 真人四川话锚点 0.0767 |
| 交叉 ASR 相对改善（仅辅助观察，见评测文档 §2.6） | 微调音频 27.3%，标准朗读 20.0% |

## 目录

```
docs/survey.md                 语音大模型选型调研（为什么不用 7B Omni）
docs/training.md               两段训练的配方与坑
docs/evaluation.md             评测口径、失败过的指标、最终数字
docs/incident-transformers5.md 事故复盘：transformers 5.17 让模型说乱码
src/                           数据准备、训练、评测、服务全部脚本
conf/                          训练配置
results/                       评测原始输出（JSON）
samples/                       demo 音频 + 回读转写（先听这里）
```

## 快速上手

### 1. 环境

沿用机器上已有的 `~/audio/venv`（torch 2.5.1+cu124），**不重装 torch**。
CosyVoice 源码在 `~/CosyVoice`，需要打一个兼容补丁（见事故复盘）：

```bash
# cosyvoice/llm/llm.py 的 forward_one_step 里按 cache 长度重建 attention mask
bash src/patch_transformers5.sh ~/CosyVoice
```

不打这个补丁，在 transformers ≥5 上原版 CosyVoice2 会输出乱码（实测 33 字句子生成 25.6 秒
的噪声）。这是本仓库最重要的一条坑。

### 2. 准备数据

```bash
python src/prep_all2.py          # 生成 wav.scp/text/utt2spk/spk2utt
bash  src/prep_v2.sh             # 抽说话人向量 + speech token + 打 parquet
```

### 3. 训练

```bash
bash src/train_llm2.sh           # TTS（CosyVoice2 LLM SFT，单卡 fp32，约 16 GB，2.5 min/epoch）
python src/train_asr2.py         # ASR（SenseVoiceSmall 全参，约 5.3 GB，4 epoch 约 10 min）
```

两个任务都只占 GPU1，GPU0 留给 ComfyUI。

### 4. 评测

```bash
python src/eval_asr.py    --ckpts ~/chuan/models/sensevoice-sichuan.pt
python src/eval_tts_final.py --sft_dir ~/chuan/models/cosyvoice2-sichuan2
python src/eval_tts_cross.py     # base vs 川话微调 ASR 交叉判读
python src/check_speaker_sim.py  # 音色保真度
python src/sweep_epochs.py       # 检查点扫描
```

### 5. 起服务

```bash
systemctl --user enable --now sichuan-audio.service   # 端口 8100，只绑 GPU1
```

```bash
curl -X POST localhost:8100/tts  -F "text=今天天气好安逸哦，我们出去耍一哈嘛。" -F "voice=sichuan" -o out.wav
curl -X POST localhost:8100/asr  -F "file=@out.wav"                   # 川话识别器
curl -X POST localhost:8100/asr  -F "file=@out.wav" -F "model=base"   # 原版普通话识别器
curl localhost:8100/voices
```

**best-of-N**：TTS 采样是随机的，同一句话重采样会得到不同读音（实测「归一」首采正确率
`sichuan` 2/8、`sichuan_avg` 3/8）。`/tts` 的 `n_samples` 参数（默认 5，可设 1–5）会合成
多个候选，用川话 ASR 回读算 CER，返回最好的一个；命中 CER 0 就提前停。响应头里的
`X-Best-Cer` 是选中候选的回读 CER，`X-Samples` 是实际用了几个候选。代价是延迟乘以
候选数——4 秒音频 best-of-5 约 5–6 秒。

一键验收（合成 → 双识别器回读 → 算 CER，三个音色都跑一遍）：

```bash
python3 src/verify_service.py                 # 默认打 http://192.168.3.178:8100
python3 src/verify_service.py http://localhost:8100
```

通过时输出 `RESULT: PASS`，退出码 0。批量生成 + 逐句回读报告用 `src/gen_demo_batch.py`
（产物格式见 `samples/batch-20261030/`），采样方差探测用 `src/variance_probe.py`。

四种声音：

| voice | 引擎 | 说明 |
|---|---|---|
| `sichuan` | 微调模型 | 成都女声，无需提示音频，**默认** |
| `sichuan_avg` | 微调模型 | 2233 条川话说话人向量均值（平均音色） |
| `sichuan_zeroshot` | 原版模型 | 川话提示音零样本克隆，对照用，稳定性差 |
| `mandarin` | 原版模型 | 普通话女声，原能力保留 |

## 模型权重

GitHub 放不下 2 GB 的 `llm.pt`。权重在训练机的这两个路径：

```
~/chuan/models/cosyvoice2-sichuan2/    # TTS：llm.pt（epoch_4）+ spk2info.pt + 指向基座的软链
~/chuan/models/sensevoice-sichuan.pt   # ASR：SenseVoiceSmall 微调权重（epoch_2）
```

`src/build_sft_model2.py` 可以从任意 SFT checkpoint 重建一个可推理的模型目录。

## 数据来源

全部为公开可下载语料：

| 语料 | 条数 | 用途 |
|---|---|---|
| MagicHub 四川话 TTS lite（成都女声） | 78 | 川话训练 + 提示音 |
| liugztat Southwestern Mandarin | 1962 | 川话训练 |
| KeSpeech 西南官话（test split） | 2293 | 川话训练（与 400 条评测集不相交） |
| THCHS-30 普通话 | 900 | 防遗忘回放 |

合计 5046 条 / 9.12 小时，其中川话 7.6 小时。构建脚本里写死了 HF-mirror 路径。

> 方法学妥协：KeSpeech 官方只公开了 test split，train split 需要申请，所以训练用了 test split
> 的一部分、评测用另一部分（按 utt id 显式剔除）。评测集与训练集不相交，但严格说不是官方
> train/test 划分。这一点在 `docs/evaluation.md` 里也写明了。

## 已知限制

1. **方言"像不像"没有公认客观指标**。本仓库用 ASR-CER 对照 + 交叉识别 + 声学统计三条独立证据，
   都指向同一结论，但最终"像不像四川人说话"仍需人耳判断。试过方言分类器（说话人无关准确率
   0.567）和 s/sh 平翘舌声学检验（对齐粒度 56 ms 太粗），两个指标都失败，失败记录留在
   `results/` 与 `docs/evaluation.md`。
2. **TTS 只微调了 LLM**，flow 与声码器保持原样。口音由 LLM 的 text→speech-token 映射决定，
   音色由说话人向量控制（CosyVoice2 的 LLM 不使用说话人向量）。
3. **词级读音错误仍存在，且分两类**：
   - 随机型（「归一」读成「鬼一/诡一/归你」，首采正确率 25%–37%）——`n_samples` 的
     best-of-N 能把成功率抬到 76%–90%，但不能根治。
   - 系统型（「收摊摊」读成「说摊塌/海滩儿」、「切」按普通话读成 qù）——叠词名词和
     方言词读音是训练语料的长尾，best-of-N 也救不了，只能靠加数据。
   逐句证据见 `samples/batch-20261030/README.md`。
4. 语料以成都/重庆片为主，川南、川东口音覆盖不足。
5. 9.12 小时属于小样本领域迁移，继续加数据仍有提升空间。
