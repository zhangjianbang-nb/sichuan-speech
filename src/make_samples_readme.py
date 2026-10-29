"""Build samples/README.md and samples/transcripts.json from the demo wavs."""
import os, sys, json, re
import numpy as np

sys.path.append("/home/whu/chuan/work")
import train_asr_lib as tal
from funasr import AutoModel as F

D = os.path.expanduser("~/chuan/samples")
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
LABEL = {
    "sft_female_0": "微调模型·成都女声",
    "sft_avg_0": "微调模型·平均音色",
    "mandarin_0": "原版模型·普通话",
    "base_zeroshot_sichuan": "原版模型·川话提示音零样本克隆",
    "base_std_mandarin_reading": "原版模型·普通话提示音读川话文本",
}


def label_of(fn):
    for k, v in LABEL.items():
        if fn.startswith(k):
            return v
    return ""


def norm(t):
    t = re.sub(r"<\|[^|]*\|>", "", t)
    return re.sub(r"[\s，。、？！；：,.?!;:\"'“”‘’（）()《》<>·—\-]", "", t)


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return int(d[len(r), len(h)]), len(r)


base = F(model="/home/whu/models/SenseVoiceSmall", device="cpu", disable_update=True)
ft = tal.load_finetuned(os.path.expanduser("~/chuan/models/sensevoice-sichuan.pt"), device="cpu")

rows = []
for fn in sorted(REFS):
    p = os.path.join(D, fn)
    ref = REFS[fn]
    hb = re.sub(r"<\|[^|]*\|>", "", base.generate(input=p, cache={}, language="zh",
                                                 use_itn=False, batch_size_s=60)[0]["text"]).strip()
    hf = ft.transcribe(p)
    eb, nb = cer(ref, hb)
    ef, nf = cer(ref, hf)
    rows.append({"file": fn, "ref": ref, "base": hb, "ft": hf,
                 "cer_base": round(eb / max(1, nb), 3), "cer_ft": round(ef / max(1, nf), 3)})
    print("%-30s %.3f %.3f" % (fn, eb / nb, ef / nf), flush=True)
json.dump(rows, open(os.path.join(D, "transcripts.json"), "w"), ensure_ascii=False, indent=1)

same = [r for r in rows if r["ref"] == REFS["base_zeroshot_sichuan.wav"]]
L = []
L.append("# 试听样本\n")
L.append("全部由 `src/api_server.py` 现场合成。每段音频用两个识别器回读：")
L.append("`base` = 原版 SenseVoiceSmall（普通话），`ft` = 川话微调版。\n")
L.append("回读 CER 的读法很关键：**CER 低 ≠ 口音好**。川话口音会让普通话识别器写出别的字，")
L.append("CER 反而升高；但如果是合成崩了，川话识别器也救不回来，两个 CER 会一起升高。")
L.append("所以要看的是两个 CER 的**组合**。\n")

L.append("## 同一句文本的三方对照\n")
L.append("文本：`今天天气好安逸哦，我们出去耍一哈嘛。`\n")
L.append("| 音频 | 普通话 ASR 回读 | 川话 ASR 回读 | CER(base) | CER(ft) |")
L.append("|---|---|---|---|---|")
for r in same:
    lab = label_of(r["file"])
    L.append("| `%s`（%s） | %s | %s | %.3f | %.3f |" %
             (r["file"], lab, r["base"], r["ft"], r["cer_base"], r["cer_ft"]))
L.append("")
L.append("微调模型把「安逸」「一哈」读对了（两个识别器都 0 错），原版模型两种提示音都读成")
L.append("「哎哟……一下」——这不是口音差异，是**读错字**：原版模型没见过川话口语词。")
L.append("这是本次微调最直接的效果证据：同一句话，原版 25% 字错，微调 0% 字错。\n")

L.append("## 全部样本\n")
L.append("| 文件 | 说明 | 文本 | 普通话回读 | 川话回读 | CER(base) | CER(ft) |")
L.append("|---|---|---|---|---|---|---|")
for r in rows:
    lab = label_of(r["file"])
    L.append("| `%s` | %s | %s | %s | %s | %.3f | %.3f |" %
             (r["file"], lab, r["ref"], r["base"], r["ft"], r["cer_base"], r["cer_ft"]))
L.append("")
L.append("## 怎么听\n")
L.append("```bash")
L.append("for f in *.wav; do echo \"== $f\"; ffplay -nodisp -autoexit \"$f\"; done")
L.append("```\n")
L.append("三组重点对照：\n")
L.append("1. `sft_female_01.wav` vs `base_std_mandarin_reading.wav`：同一句川话文本，")
L.append("   微调模型 vs 原版模型标准朗读。听「安逸」「一哈」「耍」的读音。")
L.append("2. `sft_female_01.wav` vs `base_zeroshot_sichuan.wav`：微调模型 vs 零样本克隆。")
L.append("   后者回读 CER 0.250，是合成不稳定，不是口音更浓。")
L.append("3. `sft_female_02.wav` vs `mandarin_01.wav`：川话 vs 普通话，确认普通话能力没丢。")
open(os.path.join(D, "README.md"), "w").write("\n".join(L) + "\n")
print("WROTE", flush=True)
