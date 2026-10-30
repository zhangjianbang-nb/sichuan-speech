# 四川方言语音模型：让 CosyVoice2 会说川话、让 SenseVoice 听得懂川话

[![CI](https://github.com/zhangjianbang-nb/sichuan-speech/actions/workflows/ci.yml/badge.svg)](https://github.com/zhangjianbang-nb/sichuan-speech/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

在单张 RTX 4090（24 GB）上，用 9.12 小时开源川话语料，把普通话 TTS 和 ASR 迁移成四川方言，
并保持普通话能力不退化。仓库包含完整数据流水线、训练与评测脚本、踩坑记录、可直接调用的
HTTP 服务，以及一个不依赖任何 CDN 的浏览器界面。

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

原版模型把「安逸」「一哈」读成「哎哟」「一下」——**读错字**，不是口音差异；微调模型两个
识别器都 0 错。这一句是单句示例；20 句批量上微调的平均 CER 是 0.0483、原版 0.0552，
优势没有单句这么夸张（见 `docs/evaluation.md` §2.2）。

**其他证据**

| 指标 | 结果 |
|---|---|
| 音色相似度（campplus 余弦） | 0.678（目标说话人 vs 合成音频） |
| 普通话 TTS 回归 | 4 句普通话文本 CER 0.0000 |
| 合成川话口音强度（普通话 ASR CER） | 0.0748 vs 真人四川话锚点 0.0767 |
| 交叉 ASR 相对改善（仅辅助观察，见评测文档 §2.6） | 微调音频 27.3%，标准朗读 20.0% |

## 快速开始

```bash
pip install "sichuan-speech[serve]"

export SICHUAN_COSYVOICE_HOME=~/CosyVoice
export SICHUAN_BASE_DIR=~/models/CosyVoice2-0.5B
export SICHUAN_SFT_DIR=~/chuan/models/cosyvoice2-sichuan2
export SICHUAN_ASR_BASE_DIR=~/models/SenseVoiceSmall
export SICHUAN_ASR_CKPT=~/chuan/models/sensevoice-sichuan.pt
export SICHUAN_PROMPT_WAV=~/chuan/data/raw/magic_tts/1776909066881_g4plqjyx_32946.wav

sichuan-speech serve            # http://127.0.0.1:8100/ 就是界面
sichuan-speech info             # 体检：哪些路径不存在
sichuan-speech verify           # 端到端验收：3 个声音各一句，双识别器回读
```

命令行一次合成：

```bash
sichuan-speech tts "今天天气好安逸哦，我们出去耍一哈嘛。" --voice sichuan --n-samples 5 -o out.wav
sichuan-speech asr out.wav --model ft
```

HTTP 直接调：

```bash
curl -X POST localhost:8100/tts -F "text=今天天气好安逸哦" -F "voice=sichuan" -o out.wav
curl -X POST localhost:8100/asr -F "file=@out.wav"                  # 川话识别器
curl -X POST localhost:8100/asr -F "file=@out.wav" -F "model=base"  # 原版普通话
curl localhost:8100/voices
```

> **必读**：在 transformers ≥ 5 上，原版 CosyVoice2 会输出乱码（33 字句子生成 25.6 秒噪声），
> 因为缓存解码时的 attention mask 被按字面展开。先打补丁：`bash scripts/patch_transformers5.sh`。
> 事故取证见 `docs/incident-transformers5.md`。

## best-of-N：用采样换准确率

TTS 采样是随机的，同一句话重采样读音不同（实测「归一」首采正确率 25%–37%）。`/tts` 的
`n_samples`（默认 5，可设 1–5）会渲染多个候选，用川话识别器逐条回读算 CER，返回最低的一条；
命中 0 立即停止。响应头 `X-Best-Cer` 是选中候选的 CER，`X-Samples` 是**实际渲染条数**。
代价是延迟乘以候选数——4 秒音频 best-of-5 约 5–6 秒，命中率抬到 76%–90%。

## 四种声音

| voice | 引擎 | 说明 |
|---|---|---|
| `sichuan` | 微调模型 | 成都女声，无需提示音频，**默认** |
| `sichuan_avg` | 微调模型 | 2233 条川话说话人向量均值（平均音色） |
| `sichuan_zeroshot` | 原版模型 | 川话提示音零样本克隆，对照用，稳定性差 |
| `mandarin` | 原版模型 | 普通话女声，原能力保留 |

## 目录结构

```
src/sichuan_speech/       可安装的包
  config.py               环境变量驱动的配置（SICHUAN_*）
  text.py                 CER / 文本归一化（所有评测数字的口径）
  audio.py                解码、重采样、WAV 读写、拼接
  asr.py                  SenseVoice 识别器（原版 / 微调）
  tts.py                  CosyVoice2 引擎与声音注册表
  service.py              FastAPI 应用（懒加载、best-of-N、串行锁）
  client.py               纯标准库 HTTP 客户端
  verify.py / batch.py / variance.py / cli.py
  data/                   语料抽取、清单构建、Kaldi 目录
  training/               TTS / ASR / 方言分类器训练
  evaluation/             ASR、TTS、口音、音色、样本回读
  web/                    前端（index.html / app.js / style.css）
scripts/                  训练与评测的薄包装（保留原命令行用法）
docs/                     数据收集、训练、评测、推理、踩坑全记录
conf/                     训练配置
results/                  评测原始输出（JSON）
samples/                  demo 音频 + 回读转写（先听这里）
deploy/                   systemd 用户服务
tests/                    单元测试（无 GPU、无模型依赖）
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/data-collection.md](docs/data-collection.md) | 语料来源、许可证、下载、清洗规则、为什么弃用 730 条 |
| [docs/training.md](docs/training.md) | 从数据到权重的完整流程、配置、CV 曲线、耗时 |
| [docs/training-notes.md](docs/training-notes.md) | 训练经验与 18 个 bug 的「现象→根因→修法→教训」 |
| [docs/evaluation.md](docs/evaluation.md) | 评测口径、全部数字、两个失败的指标 |
| [docs/inference.md](docs/inference.md) | 安装、配置表、接口、前端、systemd、排障 |
| [docs/survey.md](docs/survey.md) | 选型调研：为什么不用 7B Omni 模型 |
| [docs/incident-transformers5.md](docs/incident-transformers5.md) | 乱码事故复盘 |

## 模型权重

GitHub 放不下 2 GB 的 `llm.pt`。权重在训练机的这两个路径：

```
~/chuan/models/cosyvoice2-sichuan2/    # TTS：llm.pt（epoch_4）+ spk2info.pt + 指向基座的软链
~/chuan/models/sensevoice-sichuan.pt   # ASR：SenseVoiceSmall 微调权重（epoch_2）
```

`python scripts/build_sft_model.py --ckpt <epoch_N_whole.pt> --out <dir>` 可以从任意 SFT
checkpoint 重建一个可推理的模型目录。

## 数据来源

全部为公开可下载语料，详见 [docs/data-collection.md](docs/data-collection.md)：

| 语料 | 条数 | 用途 |
|---|---|---|
| MagicHub 四川话 TTS lite（成都女声） | 78 | 川话训练 + 提示音 |
| liugztat Southwestern Mandarin（MIT） | 1962 | 川话训练（清洗后 1775） |
| KeSpeech 西南官话（test split） | 2293 | 川话训练（与 400 条评测集不相交） |
| THCHS-30 普通话 | 900 | 防遗忘回放（18%） |

合计 5046 条 / 9.12 小时，其中川话 7.6 小时。

> 方法学妥协：KeSpeech 官方只公开 test split，train split 需要申请，所以训练用了 test split
> 的一部分、评测用另一部分（按 utt id 显式剔除）。评测集与训练集不相交，但严格说不是官方
> train/test 划分。KeSpeech 的 HF 镜像未标注许可证，按研究用途使用；商用前须向官方确认授权。

## 已知限制

1. **方言「像不像」没有公认客观指标**。本仓库用 ASR-CER 对照 + 交叉识别 + 声学统计三条独立
   证据，都指向同一结论，但最终「像不像四川人说话」仍需人耳判断。试过方言分类器（说话人无关
   准确率 0.567）和 s/sh 平翘舌声学检验（对齐粒度 56 ms 太粗），两个指标都失败，失败记录留在
   `results/` 与 `docs/evaluation.md`。
2. **TTS 只微调了 LLM**，flow 与声码器保持原样。口音由 LLM 的 text→speech-token 映射决定，
   音色由说话人向量控制（CosyVoice2 的 LLM 不使用说话人向量）。
3. **词级读音错误仍存在，且分两类**：
   - 随机型（「归一」读成「鬼一/诡一/归你」，首采正确率 25%–37%）——best-of-N 能抬到
     76%–90%，但不能根治。
   - 系统型（「收摊摊」读成「说摊塌/海滩儿」、「切」按普通话读成 qù）——叠词名词和方言词
     读音是训练语料的长尾，best-of-N 也救不了，只能靠加数据。
   逐句证据见 `samples/batch-20261030/README.md`。
4. 语料以成都/重庆片为主，川南、川东口音覆盖不足。
5. 9.12 小时属于小样本领域迁移，继续加数据仍有提升空间。

## 开发

```bash
pip install -e ".[serve,dev]"
make lint          # ruff
make test          # pytest（无 GPU、无模型）
make build         # wheel + sdist，并检查 web/ 与 py.typed 是否打进去
```

CI（GitHub Actions）在 Python 3.10 / 3.11 / 3.12 上跑 ruff + mypy + pytest，并断言 wheel
里确实包含前端文件。

## 许可证

代码 MIT，见 [LICENSE](LICENSE)。模型权重与语料遵循各自的许可证：CosyVoice2、SenseVoice
来自 FunAudioLLM，语料许可见上表与 [docs/data-collection.md](docs/data-collection.md)。
