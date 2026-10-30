# 训练经验与 bug 全记录

按时间顺序记录这次方言微调里真正踩过的坑。每条都写清**现象 → 根因 → 修法 → 教训**，
方便下次直接查，不用重新试错。

环境：单卡 RTX 4090（GPU1）、CosyVoice2-0.5B + SenseVoiceSmall、8 小时预算、
数据 5046 条 / 9.12 小时。

## 0. 一页速查表

| # | 坑 | 一句话修法 |
|---|---|---|
| 1 | bf16 权重 + `--use_amp` 崩 | 载入后 `model = model.float()`，去掉 `--use_amp` |
| 2 | lr 1e-5 一轮就过拟合 | 降到 3e-6，扩数据到 9.12 h |
| 3 | `torchrun` 不在 PATH | 用 venv 里的绝对路径 |
| 4 | ssh heredoc 引号被吃掉 | 脚本本地写好再 scp |
| 5 | `pkill -f train_asr.py` 杀掉自己 | 用精确 pid 或更长的区分串 |
| 6 | transformers≥5 语音变乱码 | 重建全长 attention mask（`scripts/patch_transformers5.sh`） |
| 7 | `KeyError: 'embedding'` | 把 `llm_embedding` 复制到 `embedding` |
| 8 | 换说话人向量口音不变 | 正常：LLM 不吃说话人向量 |
| 9 | 情感头学到 blank | 情感位填 `25004` 不是 `0` |
| 10 | 第一轮 CER「提升」是假的 | 两条链路统一 `use_itn=False` |
| 11 | 域内 dev 好、留出集更差 | 结论只认跨说话人留出集 |
| 12 | ASR 显存 OOM | `expandable_segments` + batch 3 / 10 s |
| 13 | 自动标注语料教错发音 | 整批弃用，换人工转写语料 |
| 14 | 方言分类器 0.567 | 学到的是说话人/通道，不可用 |
| 15 | 平翘舌检验不可用 | CTC 帧长 56 ms 太粗 |
| 16 | KeSpeech 音频被数两遍 | 按 ID 去重 + 优先完整 8 分片 |
| 17 | 单次采样方言词命中率低 | best-of-N + 回读选优 |
| 18 | 按 CER 选不出「口音最浓」的检查点 | 用 CV loss 选，CER 只作报告 |

## 1. bf16 预训练权重 + `--use_amp` 直接崩

**现象**：

```
RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
```

**根因**：CosyVoice2-0.5B 的 `CosyVoice-BlankEN/config.json` 里 `torch_dtype` 是 `bfloat16`。
GradScaler 的 unscale 算子没有 bf16 实现，`--use_amp` 一开就撞上。

**修法**：在 `train.py` 载入 checkpoint 之后加 `model = model.float()`，并且**去掉 `--use_amp`**。
fp32 在 4090 上单卡完全放得下（16 GB）。

**教训**：混合精度不是默认正确的。拿到一个「别人能跑」的权重，先看它的 dtype，
再看训练脚本对 dtype 做了什么假设。

## 2. 学习率 1e-5：一轮就过拟合

**现象**：第一轮 6.08 小时数据、lr 1e-5。CV loss 从 epoch 0 的 3.1697 只降到 epoch 1 的
3.1344，之后一路涨到 epoch 7 的 4.5628——有效学习只有一轮。

**根因**：0.5B 模型 + 6 小时数据 + 1e-5，参数量相对数据量太大。

**修法**：lr 降到 3e-6，语料扩到 9.12 小时并剔除噪声标签，得到 9 轮可用曲线（epoch 4 触底）。

**教训**：小数据全参微调，先按「数据小时数」估学习率量级。1e-5 是官方 LibriTTS（几百小时）
的量级，不是 6 小时的量级。

## 3. `torchrun` 不在 PATH

**现象**：`bash: torchrun: command not found`，但 `python -c "import torch"` 正常。

**根因**：venv 里没有 `pip` 可执行文件（只有 `python -m pip`），`torchrun` 脚本也没进 PATH。

**修法**：用绝对路径 `/home/whu/audio/venv/bin/torchrun`；新脚本里用 `SICHUAN_TORCHRUN`
覆盖，默认 `torchrun`。

**教训**：远程环境先 `which torchrun python`，别假设 PATH 和本地一致。

## 4. ssh heredoc 里的引号被外层 shell 吃掉

**现象**：`ssh 'python3 - <<EOF ... EOF'` 里嵌单引号，远程收到的脚本缺字符、语法错。

**根因**：单引号包裹的整段命令里再出现单引号，会提前结束引用。

**修法**：远程脚本一律本地写好再 `scp`，或 `ssh bash -s < script.sh`。

**教训**：跨层引号嵌套是不可靠的，能落文件就落文件。

## 5. `pkill -f "train_asr.py"` 杀掉自己的 ssh 会话

**现象**：`pkill` 执行后 ssh 立刻断，exit 255。

**根因**：ssh 命令行本身含有 `train_asr.py` 字符串，`pkill -f` 匹配到了自己的 shell。

**修法**：用精确 pid（`pgrep -f` 先看，再 `kill <pid>`），或给区分串（完整路径 + 参数）。

**教训**：`pkill -f` 是「匹配整条命令行」，不是「匹配进程名」。

## 6. transformers ≥ 5：语音变乱码

**现象**：推理输出是一串无意义音节，ASR 回读 CER 接近 1；换成 transformers 4.x 就正常。

**根因**：`cosyvoice/llm/llm.py` 的 `forward_one_step` 在缓存解码时传入长度 1 的
`attention_mask`，而 KV cache 里已经有完整 prompt。transformers ≥ 5 会按字面展开这个 mask，
把历史位置全部屏蔽，模型只能看到 SOS token。

**修法**：在 `forward_one_step` 里按 `cache.get_seq_length()` 重建全长 mask。
已固化为 `scripts/patch_transformers5.sh`，幂等、可重复执行。

**教训**：跨大版本升级时，先跑一条最短的端到端用例（一句合成 + 回读），比读 changelog 快。
现场取证见 [incident-transformers5.md](incident-transformers5.md)。

## 7. `KeyError: 'embedding'`：spk2info schema 不一致

**现象**：`inference_sft` 报 `KeyError: 'embedding'`，但 `spk2info.pt` 里明明有这个说话人。

**根因**：`AutoModel.add_zero_shot_spk()` 存的是 `llm_embedding` / `flow_embedding`，
而 `frontend_sft()` 读的是 `spk2info[spk_id]['embedding']`。两个函数对 schema 的假设不一致。

**修法**：`build_inference_model()` 里把 `llm_embedding` 复制一份到 `embedding`。

**教训**：同一个数据结构被两个模块读写时，去读两边的代码，不要相信字段名。

## 8. 换说话人向量，口音却不变

**现象**：`sichuan_female` 和 `sichuan_avg` 音色明显不同，但回读转写一模一样。

**根因**：读 `Qwen2LM.inference` 可以看到 `lm_input` 只由
`[sos, text_emb, task_id, prompt_speech_token]` 拼成，**没有 embedding 项**。

**修法**：不需要修，这是模型结构决定的。文档里写清楚：**口音来自 LLM，音色来自 flow**。

**教训**：解释一个现象前，先确认它是不是「本该如此」。这条结论直接决定了后续所有实验设计
（口音迁移只需训 LLM）。

## 9. 情感 token 填 0，情感头学到 blank

**现象**：第一轮 ASR 微调后，输出头的情感/事件位异常，CER 也比预期差。

**根因**：SenseVoice 的标签前缀是 `[<|zh|>, <|Speech|>, <|NEUTRAL|>, <|woitn|>]`，
第一轮把情感位填成了 `0`（blank），等于教模型「这里应该是空白」。

**修法**：填 `25004`（`<|NEUTRAL|>`）。`label_prefix()` 里已经写死并加了注释。

**教训**：CTC 的 blank 是「没有输出」，不是「中性」。特殊 token 区（24884 起）不能随便填 0。

## 10. 第一轮的「提升」是假的：ITN 口径不一致

**现象**：第一轮报「西南官话 0.1394 → 0.1258，普通话 0.0685 → 0.0449」，看起来很好。

**根因**：基线用 `use_itn=True`、微调版用 `use_itn=False`。基线把数字读成汉字
（「七千多种」→「7000多种」），参考文本是汉字，于是基线凭空多出一批错误。

**修法**：统一 `use_itn=False`。基线真实值是 0.1078 / 0.0384，第一轮微调
（0.1258 / 0.0449）其实**比基线更差**。整个第一轮结论作废，重训第二轮。

**教训**：任何对比评测，两条链路的口径必须逐字对齐，尤其是「看起来无关」的后处理开关。
这是本次最贵的一个错误，浪费了一整轮训练和评测。

## 11. 域内 dev 好，跨说话人留出集更差

**现象**：第一轮 lr 2e-5 训 8 轮，域内 DEV_CER 最低 0.0773（epoch 4），但跨说话人留出集上
川话 0.1258 vs 基线 0.1078、普通话 0.0449 vs 0.0384——两个方向都更差。

**根因**：域内 dev 与训练集同说话人、同录音环境，过拟合被掩盖了。

**修法**：结论只认跨说话人留出集；域内 dev 仅用于选 epoch。

**教训**：**域内 dev 只能用来选 epoch，不能用来下结论。** 留出集必须换说话人。

## 12. ASR 显存 OOM

**现象**：TTS 训练同时占着 GPU 时，ASR batch 4 / 20 秒要 5.77 GB 会 OOM；降到 batch 3 /
12 秒仍要 5.79 GB；最后 batch 1 / 8 秒稳定在 5.3 GB。

**根因**：长序列 + 碎片化。

**修法**：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，第二轮 batch 3 / 10 秒稳定
5.8 GB 不 OOM。

**教训**：`expandable_segments` 对「大 batch 短序列 / 小 batch 长序列」的碎片问题很有效，
先开它再降 batch。

## 13. 自动标注语料教错发音

**现象**：用 SenseVoice 自动标注 730 条无文本川话，抽检发现「医院」被标成「医万」、
「珍溪」被标成「真溪」。

**根因**：TTS 训练是「文本 → 语音 token」的映射，错误文本会直接教错发音。

**修法**：整批剔除，换成 KeSpeech 西南官话（人工转写）。

**教训**：**数据质量 > 数据数量。** 自动标注可以用来看趋势，不能用来训发音。

## 14. 方言分类器：0.567（不可用）

**现象**：MFCC 统计特征 + 逻辑回归，说话人无关 GroupKFold 准确率只有 0.567（随机 0.5）。
不做说话人分组时能到 0.9+。

**根因**：不做分组时，分类器学的是说话人/录音通道，不是方言。跨语料测试里它把 KeSpeech
普通话判成 97.7% 四川话，正是通道指纹泄漏。

**修法**：从评测里剔除，脚本保留为失败记录。

**教训**：**分类器类指标必须先做说话人分组交叉验证**，否则高分基本是泄漏。

## 15. 平翘舌（s/sh）声学检验：不可用

**现象**：测擦音频谱重心，真人四川话差距 +133 Hz、真人普通话 +67 Hz——与语言学预期相反。

**根因**：SenseVoice CTC 帧长 56 ms，擦音只有 60–100 ms，起始点量化误差足以把窗口推进元音里，
元音能量盖掉擦音差异。另外绕口令让 SFT 模型出现重复/漏字，音频本身不可测。

**修法**：标记不可用，数据留在 `results/minshi_failed.json`。要做可靠检验需要 10 ms 级
强制对齐（Kaldi/MFA），本仓库没有这个工具链。

**教训**：**不做没有把握的结论。** 测不准就说测不准，比编一个数字强。

## 16. KeSpeech 音频被数两遍（潜在 bug）

**现象**：旧脚本用 `glob("test-*.parquet")`，如果目录里同时有 `test-0.parquet` 和
`test-*-of-00008.parquet`，同一批音频会被读两遍。

**根因**：通配符同时命中两种命名。

**修法**：`_kespeech_parquet_files()` 优先选完整的 8 分片；`_add_kespeech()` 再用 `seen`
集合按 `ID` 去重。双保险。

**教训**：数据管道的「条数对不上」大多来自重复计数，去重要按业务主键（这里是 ID），
不要按文件名。

## 17. 单次采样命中率低 → best-of-N

**现象**：像「归一」这种难词，单次采样读对的概率只有 25–37%；5 条候选里至少一条读对的概率
升到 76–90%。

**根因**：LLM 采样是随机的，方言词的 token 概率不够集中。

**修法**：服务端渲染 N 条候选，用方言识别器逐条回读，取 CER 最低的一条；遇到 0 立即停止。
默认 N=5，接口可调 1–5。

**教训**：生成式任务的质量可以用「采样 + 判别」换，代价是 N 倍延迟。先量化单次命中率，
再决定 N。

## 18. 用 CER 选不出「口音最浓」的检查点

**现象**：20 句批量里，epoch 0–8 的普通话 CER 在 0.038–0.069 之间来回跳，不单调。

**根因**：量级（0.01–0.03）≈ 2–5 个字，属于 20 句样本下的采样噪声。

**修法**：用 CV loss 选 epoch 4；CER 只作报告，不作选择依据。

**教训**：**别用一个噪声大于信号的指标做决策。** 先估指标的置信区间，再决定它能用来干什么。

## 经验总结（可迁移的部分）

1. **先定口径，再跑实验**。ITN 那次错误说明：口径不统一，实验白跑。
2. **留出集要换说话人**。域内 dev 会骗人。
3. **同 vocoder 内比较**。所有 TTS 对照只换 LLM 和说话人向量，避免声码器差异混进结论。
4. **多条独立证据交叉验证**。ASR CER、声学统计、音色余弦、真人锚点，任何一条都能单独证伪。
5. **测不准就写「不可用」**。分类器和 s/sh 检验都保留为失败记录，不粉饰。
6. **数据质量优先**。730 条自动标注不如 2293 条人工转写。
7. **小数据低学习率**。1e-5 → 3e-6 是这次最直接的收益。
8. **生成质量可以用采样换**。best-of-N 把难词命中率从 25–37% 提到 76–90%。

## 相关文档

- [data-collection.md](data-collection.md)：语料来源、清洗规则、弃用原因
- [training.md](training.md)：完整训练流程与配置
- [evaluation.md](evaluation.md)：评测口径、数字与失败指标
- [incident-transformers5.md](incident-transformers5.md)：乱码事故的现场取证
