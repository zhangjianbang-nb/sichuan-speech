"""Sampling-variance probe.

Text-to-speech sampling is stochastic: the same sentence renders differently on
every call. This probe renders N candidates per sentence, transcribes each with
the Sichuan ASR and reports the CER distribution plus best-of-N. It is how the
"「归一」 first-sample accuracy is 25–37%, best-of-5 reaches 76–90%" claim was
measured.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .client import SpeechApiError, SpeechClient
from .text import cer

CASES: list[tuple[str, str]] = [
    ("sichuan", "老板儿，来二两小面，多放点海椒。"),
    ("sichuan", "他屋头那个猫儿又跑到房顶上去了。"),
    ("sichuan_avg", "天都快黑球了，你还不搞快点儿收摊摊？"),
    ("sichuan", "今天天气好安逸哦，我们出去耍一哈嘛。"),
    ("sichuan", "这个事情我晓得，你莫着急，慢慢来。"),
]


def run_probe(
    client: SpeechClient,
    n: int = 6,
    out_dir: str | Path = "demo/variance",
    *,
    cases: Sequence[tuple[str, str]] = tuple(CASES),
) -> dict:
    """Render ``n`` candidates per sentence and report the CER distribution."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for index, (voice, text) in enumerate(cases, start=1):
        cers: list[float] = []
        hypotheses: list[str] = []
        for sample in range(n):
            response = client.tts(text, voice, n_samples=1)
            response.save(output / f"{index}_{voice}_{sample}.wav")
            hypothesis = client.asr(
                response.audio, f"{index}_{voice}_{sample}.wav", model="ft"
            ).text
            hypotheses.append(hypothesis)
            cers.append(round(cer(text, hypothesis), 4))
        print(f"\n[{index}] {voice}  {text}")
        for sample in range(n):
            print(f"   {sample}  cer={cers[sample]:.3f}  {hypotheses[sample]}")
        best = min(cers)
        mean = sum(cers) / n
        exact = sum(1 for c in cers if c == 0.0)
        print(f"   -> best={best:.3f}  mean={mean:.3f}  exact_hits={exact}/{n}")
        results.append(
            {
                "text": text,
                "voice": voice,
                "cers": cers,
                "hyps": hypotheses,
                "best": best,
                "mean": round(mean, 4),
                "exact_hits": exact,
            }
        )

    summary = {
        "n": n,
        "mean_of_best": round(sum(r["best"] for r in results) / len(results), 4),
        "mean_of_mean": round(sum(r["mean"] for r in results) / len(results), 4),
        "mean_exact_rate": round(sum(r["exact_hits"] for r in results) / (len(results) * n), 4),
        "cases": results,
    }
    (output / "variance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"\nmean-of-best = {summary['mean_of_best']:.4f} ; "
        f"mean-of-mean = {summary['mean_of_mean']:.4f} ; "
        f"first-sample exact = {summary['mean_exact_rate']:.1%}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sichuan-speech variance",
        description="Measure TTS sampling variance and best-of-N improvement.",
    )
    parser.add_argument("n", nargs="?", type=int, default=6, help="candidates per sentence")
    parser.add_argument("--url", default="http://127.0.0.1:8100", help="service base URL")
    parser.add_argument("--out", default="demo/variance", help="output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = SpeechClient(args.url)
    try:
        run_probe(client, args.n, args.out)
    except SpeechApiError as exc:
        print(f"failed: {exc}")
        return 2
    return 0
