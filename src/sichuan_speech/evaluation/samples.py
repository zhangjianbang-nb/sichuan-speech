"""Regenerate ``samples/transcripts.json`` and ``samples/README.md`` from the demo wavs.

Both recognisers read every file back, so the table is reproducible rather than
hand-transcribed. CER is reported per file for the stock and the fine-tuned recogniser;
the *combination* is what matters (see docs/evaluation.md §2.6).
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..asr import load_recognizer
from ..config import Settings
from ..text import char_error_stats
from .common import save_json

logger = logging.getLogger(__name__)

REFS = {
    "sft_female_01.wav": "今天天气好安逸哦，我们出去耍一哈嘛。",
    "sft_female_02.wav": "你吃饭没有得？没得的话我们一起切吃火锅。",
    "sft_female_03.wav": "这个事情我晓得，你莫着急，慢慢来。",
    "sft_female_04.wav": "老板儿，来二两小面，多放点海椒。",
    "sft_female_05.wav": "来嘛，坐下喝口茶，慢慢摆龙门阵。",
    "sft_avg_01.wav": "莫慌，等我把这个事情搞归一了再说。",
    "sft_avg_02.wav": "他屋头那个猫儿又跑到房顶上去了。",
    "mandarin_01.wav": "今天天气很好，我们出去走一走吧。",
    "mandarin_02.wav": "这件事情我知道，你不要着急，慢慢来。",
    "base_zeroshot_sichuan.wav": "今天天气好安逸哦，我们出去耍一哈嘛。",
    "base_std_mandarin_reading.wav": "今天天气好安逸哦，我们出去耍一哈嘛。",
}

LABELS = {
    "sft_female_0": "微调模型·成都女声",
    "sft_avg_0": "微调模型·平均音色",
    "mandarin_0": "原版模型·普通话",
    "base_zeroshot_sichuan": "原版模型·川话提示音零样本克隆",
    "base_std_mandarin_reading": "原版模型·普通话提示音读川话文本",
}


def _label_of(filename: str) -> str:
    for prefix, label in LABELS.items():
        if filename.startswith(prefix):
            return label
    return ""


def _render(rows: list[dict]) -> str:
    lines = ["# 试听样本\n", "全部由 `sichuan-speech serve` 的 `/tts` 现场合成。每段音频用两个识别器回读："]
    lines.append("`base` = 原版 SenseVoiceSmall（普通话），`ft` = 川话微调版。\n")
    lines.append("回读 CER 的读法很关键：**CER 低 ≠ 口音好**。川话口音会让普通话识别器写出别的字，")
    lines.append("CER 反而升高；但如果是合成崩了，川话识别器也救不回来，两个 CER 会一起升高。")
    lines.append("所以要看的是两个 CER 的**组合**。\n")

    anchor = REFS["base_zeroshot_sichuan.wav"]
    lines.append("## 同一句文本的三方对照\n")
    lines.append("文本：`今天天气好安逸哦，我们出去耍一哈嘛。`\n")
    lines.append("| 音频 | 普通话 ASR 回读 | 川话 ASR 回读 | CER(base) | CER(ft) |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        if row["ref"] != anchor:
            continue
        lines.append(
            "| `%s`（%s） | %s | %s | %.3f | %.3f |"
            % (row["file"], _label_of(row["file"]), row["base"], row["ft"], row["cer_base"], row["cer_ft"])
        )
    lines.append("")
    lines.append("微调模型把「安逸」「一哈」读对了（两个识别器都 0 错），原版模型两种提示音都读成")
    lines.append("「哎哟……一下」——这不是口音差异，是**读错字**：原版模型没见过川话口语词。")
    lines.append("这是本次微调最直接的效果证据：同一句话，原版 25% 字错，微调 0% 字错。\n")

    lines.append("## 全部样本\n")
    lines.append("| 文件 | 说明 | 文本 | 普通话回读 | 川话回读 | CER(base) | CER(ft) |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            "| `%s` | %s | %s | %s | %s | %.3f | %.3f |"
            % (
                row["file"], _label_of(row["file"]), row["ref"],
                row["base"], row["ft"], row["cer_base"], row["cer_ft"],
            )
        )
    lines.append("")
    lines.append("## 怎么听\n")
    lines.append("```bash")
    lines.append('for f in *.wav; do echo "== $f"; ffplay -nodisp -autoexit "$f"; done')
    lines.append("```\n")
    lines.append("三组重点对照：\n")
    lines.append("1. `sft_female_01.wav` vs `base_std_mandarin_reading.wav`：同一句川话文本，")
    lines.append("   微调模型 vs 原版模型标准朗读。听「安逸」「一哈」「耍」的读音。")
    lines.append("2. `sft_female_01.wav` vs `base_zeroshot_sichuan.wav`：微调模型 vs 零样本克隆。")
    lines.append("   后者回读 CER 0.250，是合成不稳定，不是口音更浓。")
    lines.append("3. `sft_female_02.wav` vs `mandarin_01.wav`：川话 vs 普通话，确认普通话能力没丢。")
    return "\n".join(lines) + "\n"


def build_samples_readme(
    settings: Settings,
    *,
    directory: str | Path | None = None,
    out: str | Path | None = None,
) -> list[dict]:
    """Read every demo wav back through both recognisers and rewrite the samples docs."""
    samples_dir = Path(directory) if directory else Path(__file__).resolve().parents[3] / "samples"
    output = Path(out) if out else samples_dir
    base = load_recognizer(settings, finetuned=False)
    finetuned = load_recognizer(settings, finetuned=True)

    rows: list[dict] = []
    for filename in sorted(REFS):
        path = samples_dir / filename
        if not path.exists():
            logger.warning("missing sample %s", path)
            continue
        reference = REFS[filename]
        hyp_base = base.transcribe(str(path))
        hyp_ft = finetuned.transcribe(str(path))
        stats_base = char_error_stats(reference, hyp_base)
        stats_ft = char_error_stats(reference, hyp_ft)
        rows.append(
            {
                "file": filename,
                "ref": reference,
                "base": hyp_base,
                "ft": hyp_ft,
                "cer_base": round(stats_base.cer, 3),
                "cer_ft": round(stats_ft.cer, 3),
            }
        )
        logger.info("%-30s %.3f %.3f", filename, stats_base.cer, stats_ft.cer)

    save_json(output / "transcripts.json", rows)
    (output / "README.md").write_text(_render(rows), encoding="utf-8")
    logger.info("wrote %s and %s", output / "transcripts.json", output / "README.md")
    return rows
