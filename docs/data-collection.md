# 数据收集与清洗

本项目的方言能力全部来自公开语料，没有一条录音是现采的。本文写清每个语料的来源、
许可证、下载方式、放进训练集前做了哪些过滤，以及为什么丢掉了一整批数据。

最终训练集：**5046 条 / 9.12 小时**（train 4895 + dev 151）。
评测集：**400 条西南官话 + 300 条普通话**，与训练集按 utt id 严格不相交。

## 1. 语料清单

| 语料 | Hugging Face / 来源 | 内容 | 许可 | 用量 |
|---|---|---|---|---|
| MagicHub 四川话 TTS lite | `MagicHub/magicdata-dialect-sichuanese-tts-lite` | 成都女声，带人工文本，约 10 分钟 | 见数据集页（MagicHub 社区许可） | 78 条：提示音 + 训练 |
| liugztat 西南官话 | `liugztat/Southwestern_Mandarin_Dialectal_Expressions_Speech` | 1962 条西南官话朗读，带人工文本与方言词标注 | MIT | 1775 条进训练 |
| KeSpeech test split | `TwinkStart/KeSpeech`（HF 镜像，8 个 parquet 分片，3.3 GB） | 同一批录制里既有西南官话也有普通话，带人工文本 | HF 镜像页未标注许可证，按**研究用途**使用 | 2293 条西南官话进训练；400 条西南 + 300 条普通话留作评测 |
| THCHS-30 | 原始发布为 OpenSLR 18；本机用的是 HF 上的 parquet 版本 | 普通话朗读 | 按其发布页（OpenSLR 18） | 900 条作普通话回放，占训练集 18% |
| aotemande/sichuanhua | `aotemande/sichuanhua` | 730 条川话，**无文本** | 见数据集页 | **整批弃用**，原因见 §4 |

> 许可证提醒：KeSpeech 在 HF 镜像上没有写明许可证，公开仓库里引用它的论文都按研究用途使用。
> 如果要商用，必须回到 KeSpeech 官方渠道确认授权，不要拿本文当法律依据。

## 2. 下载与目录布局

国内网络直接连 HF 经常超时，统一走 `hf-mirror.com`：

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -U "huggingface_hub[cli]"

hf download MagicHub/magicdata-dialect-sichuanese-tts-lite --repo-type dataset \
  --local-dir "$SICHUAN_DATA_ROOT/raw/magic_tts"
hf download liugztat/Southwestern_Mandarin_Dialectal_Expressions_Speech --repo-type dataset \
  --local-dir "$SICHUAN_DATA_ROOT/raw/xinan_raw"
hf download TwinkStart/KeSpeech --repo-type dataset \
  --include "test-*-of-*.parquet" --local-dir "$SICHUAN_DATA_ROOT/raw/kespeech"
hf download THCHS-30 --repo-type dataset \
  --include "thchs_train0.parquet" --local-dir "$SICHUAN_DATA_ROOT/raw/mandarin"
```

代码期望的目录结构（`$SICHUAN_DATA_ROOT` 默认 `~/chuan/data`）：

```
raw/
  magic_tts/
    txt.txt                     # 每行「文件名<TAB>文本」
    *.wav
  xiNan/                        # liugztat 解包后的 wav
  xiNan_text.txt                # 每行「文件名<TAB>带标注文本<TAB>方言词」
  kespeech/
    test-00000-of-00008.parquet …
  kespeech_sw_train/            # 由代码从 parquet 抽出的西南官话 wav
  mandarin/
    thchs_train0.parquet
    wav/                        # 由代码从 parquet 抽出的普通话 wav
  sichuanhua/                   # 仅 round 1 使用（无文本语料）
sft/
  all_sichuan.json              # 人工文本清单（下一步生成）
  train/ dev/                   # Kaldi 目录
  all_rows_v2.json              # 完整清单
eval/
  kespeech_sw.json
  kespeech_mandarin.json
```

## 3. 清洗规则（全部可复算）

### 3.1 文本清洗

liugztat 的文本带方言词标注：`差点【<筐瓢>】`。`【】<>` 只是标注符号，词本身要保留，
所以只删符号不删内容，清洗后是 `差点筐瓢`。实现见
`src/sichuan_speech/data/sichuan.py` 的 `clean_annotation()`。

### 3.2 时长与长度过滤

所有语料统一两条：**时长 1–25 秒**、**文本 ≥ 2 字**。

- 太短（<1 s）的片段在 CosyVoice 的 tokenizer 里几乎没有语音 token，训练信号是噪声。
- 太长（>25 s）的片段在 4090 上要拆 batch，且 TTS 训练按 30 秒截断，尾部文本会丢。
- 单字文本无法体现韵律，也容易被 ASR 判成空。

### 3.3 去重

KeSpeech 的 parquet 里同一条音频可能出现在多个分片（镜像打包方式导致）。
`_add_kespeech()` 用 `seen` 集合按 `ID` 去重；这是对旧脚本的修复——旧版按
`test-*.parquet` 通配时，如果同时存在 `test-0.parquet` 和 `test-*-of-00008.parquet`，
同一批音频会被数两遍。现在的 `_kespeech_parquet_files()` 优先选完整的 8 分片。

### 3.4 留出集隔离

KeSpeech 的 400 条西南官话评测音频在构建训练清单时**按 utt id 显式剔除**，
所以评测集从未参与训练。这不是官方 train/test 划分（官方只公开了 test split），
是本仓库自己的切分，必须如实标注。

## 4. 为什么弃用 aotemande/sichuanhua（730 条）

这批音频没有文本。第一轮的做法是用 SenseVoice 自动标注，再按标注训练 TTS。抽检发现
标注错误很实在：

| 音频内容 | 自动标注 | 实际 |
|---|---|---|
| 医院 | 医**万** | 医院 |
| 珍溪 | **真**溪 | 珍溪 |

TTS 训练是「文本 → 语音 token」的映射：错误文本会直接教模型把某个词读成另一个词，
而且这种错误会覆盖正确发音。730 条里有相当比例带错，整批剔除比逐条校对便宜。
第二轮改用 KeSpeech 的西南官话（人工转写）替代，数据量更大、标签可信。

旧脚本 `label_sichuanhua()` 保留在 `src/sichuan_speech/data/sichuan.py`，只为复现 round 1。

## 5. 评测集怎么来

```bash
python scripts/extract_kespeech.py --sw-limit 400 --man-limit 300
```

- 从 KeSpeech test split 的 parquet 里按 `Dialect` 字段筛出 `Southwestern` 与普通话。
- 按 `ID` 去重后各取前 400 / 300 条，写成 `data/eval/kespeech_{sw,mandarin}.json`，
  每行 `{"wav": ..., "text": ..., "id": ...}`。
- 普通话集同时承担两个角色：ASR 的遗忘守卫、TTS 的普通话回归检查。

## 6. 生成训练清单

```bash
python scripts/prep_sichuan.py          # 生成 data/sft/all_sichuan.json（1853 条）
python scripts/prep_manifest.py         # 生成 train/dev Kaldi 目录 + all_rows_v2.json
```

`prep_manifest.py` 做的事：

1. 读 `all_sichuan.json`（缺失时自动调 `prep_sichuan.py` 的逻辑现场生成）。
2. 从 KeSpeech parquet 抽西南官话 wav 到 `raw/kespeech_sw_train/`，剔除评测 id，按 id 去重。
3. 从 THCHS-30 parquet 抽 900 条普通话（`seed=7` 固定打乱后截断）。
4. 按 `seed=1986` 打乱全部行，前 3%（至少 20 条）作 dev。
5. 写 `data/sft/{train,dev}/{wav.scp,text,utt2spk,spk2utt}` 与 `all_rows_v2.json`。

预期输出：

```
clean Sichuan: 1853 utts
KeSpeech Southwestern added: 2293
Mandarin replay: 900
TOTAL 5046 utts / 9.12 h (train 4895, dev 151)
```

`--legacy` 复现第一轮（把 730 条自动标注的川话加回去），只为对照，不要用它训练。

## 7. 数据质量自查

训练前建议跑三条检查，都是分钟级：

1. **文本抽检**：随机抽 30 条，人耳听 3–5 条，确认文本与音频对得上。自动标注语料的
   错误就是这样发现的。
2. **说话人分布**：`wc -l data/sft/train/utt2spk` 与 `awk '{print $2}' | sort -u | wc -l`，
   确认没有某个说话人占比过高（本仓库最大单说话人占 12%）。
3. **时长分布**：`python - <<'PY'` 读 `all_rows_v2.json` 统计第 5 列，确认没有 0 秒文件
   （`soundfile` 读失败会记 0.0，这些行会被过滤掉，但值得确认数量）。

## 8. 复现清单

```bash
export SICHUAN_DATA_ROOT=~/chuan/data
python scripts/extract_kespeech.py
python scripts/prep_sichuan.py
python scripts/prep_manifest.py
# 预期：5046 条 / 9.12 小时
```

训练见 [training.md](training.md)，评测口径见 [evaluation.md](evaluation.md)，
踩过的坑见 [training-notes.md](training-notes.md)。
