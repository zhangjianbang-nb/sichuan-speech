# 语音大模型选型调研：既要会听（ASR）又要会说（TTS）

任务约束：单卡 RTX 4090（24 GB，实际可用 23 GB）、8 小时、要把普通话模型迁移成四川方言，
且不能破坏原有普通话能力。本文记录选型依据，以及最后为什么没选 7B 级 Omni 模型。

## 候选清单

| 模型 | 规模 | ASR | TTS | 24 GB 内可训 | 仓库自带方言/领域微调配方 | 结论 |
|---|---|---|---|---|---|---|
| Qwen2.5-Omni-7B | 7B | ✓ | ✓（Thinker-Talker 双流） | 全参不行；LoRA 可以但显存紧 | 无方言配方 | 放弃 |
| Qwen3-Omni | 更大 | ✓ | ✓ | 不行 | 无 | 放弃 |
| GLM-4-Voice | 9B | ✓ | ✓ | 推理都要多卡 | 无 | 放弃 |
| MiniCPM-o 2.6 | 8B | ✓ | ✓ | 不行 | 无 | 放弃 |
| Step-Audio 2 | 130B / mini 8B | ✓ | ✓ | mini 勉强，全参不行 | 无 | 放弃 |
| MiMo-Audio | 7B | ✓ | ✓ | 勉强 | 无 | 放弃 |
| Higgs Audio v2 | 3B | ✗ | ✓ | 可以 | 无 | 只做 TTS，不满足"会听" |
| **CosyVoice2-0.5B** | 0.5B | ✗ | ✓ | 轻松，**官方 SFT 脚本齐全** | 有（`examples/libritts/cosyvoice2`） | **选用（TTS）** |
| **SenseVoiceSmall** | 234M | ✓ | ✗ | 轻松，**官方微调脚本齐全** | 有（`finetune.sh`） | **选用（ASR）** |

## 为什么不用"一个模型全包"

7B–9B 的 Omni 模型在 24 GB 上只能 LoRA，而且：

1. **没有可抄的方言配方**。这几个仓库都只提供通用指令微调/对齐脚本，没有"给定方言语料，
   把发音和韵律迁过去"的成熟流程。8 小时内从零试错，失败概率远高于成功。
2. **TTS 侧的训练信号更脆**。Omni 模型的语音输出是 codec token，需要同时对齐文本和音频流；
   方言迁移要动的是发音层，LoRA 的低秩约束未必够。
3. **验收要求是"有效果"且"不破坏原能力"**。这两个指标都需要可控的训练量和可回退的基线，
   0.5B TTS + 234M ASR 的组合可以做到全参微调、每轮存 checkpoint、随时回退。

## 为什么选这一对

- **CosyVoice2-0.5B**：LLM（Qwen2-0.5B）+ flow matching + HiFi-GAN 声码器的三段式 TTS。
  官方给了完整的 SFT 流程（Kaldi 风格数据目录 → campplus 说话人向量 → speech tokenizer →
  parquet → `train.py --model llm`），在 4090 上单卡 fp32 训练显存约 16 GB，一轮 2–3 分钟。
- **SenseVoiceSmall**：CTC + rich-transcription 多任务 ASR，234M 参数，官方 `finetune.sh`
  可直接改数据路径。它的输出头包含语种/情感/事件标记，方言迁移只需替换文本标签。

两者拼在一个 FastAPI 后面，对外就是"一个能听会说四川话的服务"——功能上等价于一个 Omni 模型，
但每一段都可训、可评测、可回退。

## 数据可达性（本机实测）

| 语料 | 内容 | 获取方式 | 用量 |
|---|---|---|---|
| MagicHub 四川话 TTS lite | 成都女声，带文本，10 分钟 | hf-mirror | 78 条（提示音 + 训练） |
| liugztat Southwestern Mandarin | 1962 条西南官话 + 文本 | hf-mirror | 1775 条进训练 |
| aotemande/sichuanhua | 730 条川话，**无文本** | hf-mirror | **弃用**（见下） |
| KeSpeech | 西南官话 + 普通话，同批录制，带文本 | hf-mirror（8 个 test 分片 3.3 GB） | 2293 条西南官话进训练；400 条西南 + 300 条普通话留作评测 |
| THCHS-30 | 普通话朗读 | hf-mirror parquet | 900 条作回放，防遗忘 |

**aotemande/sichuanhua 为什么弃用**：这批音频没有文本，我先用 SenseVoice 自动标注，
但抽检发现标注错误明显（"要寄一县人民医万" 应为 "医院"，"真溪" 疑为 "珍溪"）。
TTS 训练是文本→语音 token 的映射，错误文本会直接教错发音，所以第二轮把它整批剔除，
换成 KeSpeech 的西南官话（人工转写）。

## 评测口径的选择（踩过的坑）

方言"像不像"很难量化。试过三条路，前两条失败：

1. **方言分类器**（mel + CNN，西南官话 vs 普通话二分类）。用 KeSpeech 同批录制的两种方言训练，
   说话人无关 5 折交叉验证只有 **0.567**（随机是 0.5）——它学到的是说话人/通道特征，不是方言。
   早期版本在跨语料测试里把 KeSpeech 普通话判成 97.7% 四川话，正是通道指纹泄漏。
   **结论：这个指标不可用，已从报告中剔除。**
2. **s/sh 平翘舌声学检验**（测擦音频谱重心差距）。CTC 强制对齐的帧长 56 ms 太粗，
   擦音只有 60–100 ms，测量误差盖过信号；真人普通话的差距反而比真人四川话更小，
   与语言学预期相反。**结论：不可用，失败数据留在 `results/minshi_failed.json`。**
3. **ASR-CER 对照**（采用）：同一套 ASR 分别作用于真实四川话锚点、真实普通话锚点、
   基线 TTS 与微调 TTS。真实四川话的 CER 天然高于普通话（0.0767 vs 0.0297，同一识别器、
   同一 `use_itn=False` 口径），这个差值就是"方言难度"的标尺；微调后的 TTS 若向四川话靠拢，
   其 CER 应向四川话锚点的量级靠拢。配合声学统计（F0 中位数/音域/下降段占比/语速）做交叉验证。

## 参考

- CosyVoice2 论文：`CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models`
- SenseVoice 论文：`FunAudioLLM: Voice Understanding and Generation Foundation Models`
- KeSpeech：`KeSpeech: An Open Source Speech Dataset of Mandarin and Its Subdialects`
