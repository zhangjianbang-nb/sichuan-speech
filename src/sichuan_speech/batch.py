"""Generate a listenable demo batch from a running service.

Each clip is synthesised with best-of-N, saved, then transcribed by BOTH
recognisers so the report shows the pair of CERs. Writes::

    <out>/NN_<voice>.wav       one clip per case
    <out>/all.wav              everything concatenated for one-shot listening
    <out>/report.json          machine-readable results
    <out>/report.md            human-readable table
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .audio import concat_wavs, write_wav
from .client import SpeechApiError, SpeechClient
from .text import cer

CASES: list[tuple[str, str]] = [
    ("sichuan", "今天天气好安逸哦，我们出去耍一哈嘛。"),
    ("sichuan", "你吃饭没有得？没得的话我们一起切吃火锅。"),
    ("sichuan", "这个事情我晓得，你莫着急，慢慢来。"),
    ("sichuan", "老板儿，来二两小面，多放点海椒。"),
    ("sichuan", "来嘛，坐下喝口茶，慢慢摆龙门阵。"),
    ("sichuan", "他屋头那个猫儿又跑到房顶上去了。"),
    ("sichuan", "我跟你说哈，昨天我在春熙路碰到个老朋友，摆了半天龙门阵，巴适得很。"),
    ("sichuan", "这个娃儿成绩好得很，考起了川大，他爸妈高兴惨了。"),
    ("sichuan_avg", "莫慌，等我把这个事情搞归一了再说。"),
    ("sichuan_avg", "天都快黑球了，你还不搞快点儿收摊摊？"),
    ("sichuan_zeroshot", "走嘛，周末我们切青城山耍，顺便吃碗担担面。"),
    ("mandarin", "今天天气很好，我们出去走一走吧。"),
]


@dataclass
class BatchRow:
    file: str
    voice: str
    text: str
    asr_ft: str
    asr_base: str
    cer_ft: float
    cer_base: float
    duration_s: float
    rtf: float
    best_cer: float | None


def run_batch(
    client: SpeechClient,
    out_dir: str | Path,
    *,
    n_samples: int = 3,
    cases: Sequence[tuple[str, str]] = tuple(CASES),
) -> dict:
    """Synthesise, verify and write the demo batch. Returns a summary dict."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[BatchRow] = []
    clips: list[Path] = []

    for index, (voice, text) in enumerate(cases, start=1):
        name = f"{index:02d}_{voice}"
        response = client.tts(text, voice, n_samples=n_samples)
        path = response.save(output / f"{name}.wav")
        clips.append(path)
        hyp_ft = client.asr(response.audio, path.name, model="ft").text
        hyp_base = client.asr(response.audio, path.name, model="base").text
        rows.append(
            BatchRow(
                file=path.name,
                voice=voice,
                text=text,
                asr_ft=hyp_ft,
                asr_base=hyp_base,
                cer_ft=cer(text, hyp_ft),
                cer_base=cer(text, hyp_base),
                duration_s=response.duration_s or 0.0,
                rtf=response.rtf or 0.0,
                best_cer=response.best_cer,
            )
        )
        print(
            f"{name:<22} dur={rows[-1].duration_s:5.2f}s rtf={rows[-1].rtf:4.1f} "
            f"cer_ft={rows[-1].cer_ft:.3f} cer_base={rows[-1].cer_base:.3f}\n"
            f"   ft  : {hyp_ft}\n   base: {hyp_base}"
        )

    combined, sample_rate = concat_wavs(clips)
    write_wav(output / "all.wav", combined, sample_rate)

    count = len(rows)
    mean_ft = sum(r.cer_ft for r in rows) / count
    mean_base = sum(r.cer_base for r in rows) / count
    dialect_rows = [r for r in rows if r.voice.startswith("sichuan")]
    dialect_ft = sum(r.cer_ft for r in dialect_rows) / len(dialect_rows)
    dialect_base = sum(r.cer_base for r in dialect_rows) / len(dialect_rows)
    exact = sum(1 for r in dialect_rows if r.cer_ft == 0.0)

    summary = {
        "n_samples": n_samples,
        "n": count,
        "mean_cer_ft": round(mean_ft, 4),
        "mean_cer_base": round(mean_base, 4),
        "dialect_mean_cer_ft": round(dialect_ft, 4),
        "dialect_mean_cer_base": round(dialect_base, 4),
        "dialect_exact_ft": exact,
        "dialect_n": len(dialect_rows),
        "rows": [
            {
                "file": r.file,
                "voice": r.voice,
                "text": r.text,
                "asr_ft": r.asr_ft,
                "asr_base": r.asr_base,
                "cer_ft": round(r.cer_ft, 4),
                "cer_base": round(r.cer_base, 4),
                "dur": round(r.duration_s, 2),
                "rtf": round(r.rtf, 2),
                "best_cer": r.best_cer,
            }
            for r in rows
        ],
    }
    (output / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    lines = [
        f"# 四川话批量合成（{count} 句，best-of-{n_samples}）",
        "",
        f"全部由 `{client.base_url}` 现场合成；每句合成 {n_samples} 个候选，按川话 ASR 回读 CER 取最好的一个。"
        "每段再用两个识别器回读：`ft` = 川话微调 SenseVoice，`base` = 原版 SenseVoiceSmall。",
        "",
        "| 文件 | 声音 | 文本 | 川话回读 | 普通话回读 | CER(ft) | CER(base) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| `{row['file']}` | {row['voice']} | {row['text']} | {row['asr_ft']} | "
            f"{row['asr_base']} | {row['cer_ft']} | {row['cer_base']} |"
        )
    lines += [
        "",
        f"平均 CER：川话识别器 {mean_ft:.4f}，普通话识别器 {mean_base:.4f}。",
        "",
        "两个识别器都 0 错 = 模型确实读对了；只有川话识别器 0 错 = 发音是方言读法，"
        "普通话识别器按普通话听写才出错（这是口音的正常表现，不是缺陷）。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        f"\nn={count}  mean cer_ft={mean_ft:.4f}  mean cer_base={mean_base:.4f}\n"
        f"Sichuan-voice clips: cer_ft={dialect_ft:.4f}  cer_base={dialect_base:.4f}  "
        f"exact(ft)={exact}/{len(dialect_rows)}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sichuan-speech batch",
        description="Generate and verify a listenable demo batch.",
    )
    parser.add_argument("out_dir", nargs="?", default="demo/batch", help="output directory")
    parser.add_argument("--url", default="http://127.0.0.1:8100", help="service base URL")
    parser.add_argument("--n-samples", type=int, default=3, help="best-of-N candidates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = SpeechClient(args.url)
    try:
        run_batch(client, args.out_dir, n_samples=args.n_samples)
    except SpeechApiError as exc:
        print(f"failed: {exc}")
        return 2
    return 0
