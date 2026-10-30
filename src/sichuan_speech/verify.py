"""Acceptance check for a running service: synthesis -> dual ASR -> CER.

Run from any machine that can reach the service::

    sichuan-speech verify http://192.168.3.178:8100

Exit code 0 means every check passed. The check is deliberately end-to-end: it
exercises the real HTTP surface (multipart, headers, both recognisers) instead of
importing the models, so it also validates the deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .client import SpeechApiError, SpeechClient
from .text import cer

DEFAULT_CASES: list[tuple[str, str]] = [
    ("sichuan", "今天天气好安逸哦，我们出去耍一哈嘛。"),
    ("sichuan_avg", "莫慌，等我把这个事情搞归一了再说。"),
    ("mandarin", "今天天气很好，我们出去走一走吧。"),
]

DIALECT_VOICES = {"sichuan", "sichuan_avg", "sichuan_zeroshot"}


@dataclass
class CheckResult:
    voice: str
    text: str
    transcript_ft: str = ""
    transcript_base: str = ""
    cer_ft: float = 1.0
    cer_base: float = 1.0
    duration_s: float | None = None
    rtf: float | None = None
    best_cer: float | None = None
    ok: bool = False
    note: str = ""


@dataclass
class VerifyReport:
    health: dict
    voices: list[str]
    checks: list[CheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(check.ok for check in self.checks) and not self.fatal

    @property
    def fatal(self) -> bool:
        return not self.health.get("ok", False)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "health": self.health,
            "voices": self.voices,
            "warnings": self.warnings,
            "checks": [
                {
                    "voice": c.voice,
                    "text": c.text,
                    "transcript_ft": c.transcript_ft,
                    "transcript_base": c.transcript_base,
                    "cer_ft": round(c.cer_ft, 4),
                    "cer_base": round(c.cer_base, 4),
                    "duration_s": c.duration_s,
                    "rtf": c.rtf,
                    "best_cer": c.best_cer,
                    "ok": c.ok,
                    "note": c.note,
                }
                for c in self.checks
            ],
        }

    def summary(self) -> str:
        lines = [
            "health: " + json.dumps(self.health, ensure_ascii=False),
            "voices: " + ", ".join(self.voices),
        ]
        for warning in self.warnings:
            lines.append(f"warning: {warning}")
        header = ["voice", "dur(s)", "rtf", "CER(ft)", "CER(base)", "best", "transcript"]
        widths = [12, 7, 6, 8, 9, 6, 40]
        lines.append("  ".join(_pad(h, w) for h, w in zip(header, widths, strict=False)))
        for check in self.checks:
            row = [
                check.voice,
                _fmt(check.duration_s),
                _fmt(check.rtf),
                f"{check.cer_ft:.3f}",
                f"{check.cer_base:.3f}",
                _fmt(check.best_cer),
                check.transcript_ft or "<failed>",
            ]
            lines.append("  ".join(_pad(v, w) for v, w in zip(row, widths, strict=False)))
            if check.note:
                lines.append(f"  ^ {check.note}")
        lines.append("")
        lines.append("RESULT: " + ("PASS" if self.ok else "FAIL"))
        return "\n".join(lines)


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in str(text))


def _pad(text: str, width: int) -> str:
    value = str(text)
    return value + " " * max(0, width - _display_width(value))


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def verify(
    client: SpeechClient,
    cases: Sequence[tuple[str, str]] = tuple(DEFAULT_CASES),
    *,
    n_samples: int = 3,
    max_cer: float = 0.25,
    expect_gpu: str = "1",
) -> VerifyReport:
    """Run the end-to-end acceptance check."""
    health = client.health()
    voice_payload = client.voices()
    voices = [v["id"] for v in voice_payload.get("voices", [])]
    report = VerifyReport(health=health, voices=voices)

    if not health.get("ok"):
        report.warnings.append("health endpoint reports not ok")
    gpu = str(health.get("gpu", ""))
    if expect_gpu and gpu != expect_gpu:
        report.warnings.append(f"service is on GPU {gpu!r}, expected {expect_gpu!r}")
    missing = {"sichuan", "mandarin"} - set(voices)
    if missing:
        report.warnings.append(f"missing voices: {', '.join(sorted(missing))}")

    for voice, text in cases:
        check = CheckResult(voice=voice, text=text)
        try:
            response = client.tts(text, voice, n_samples=n_samples)
            if not response.audio.startswith(b"RIFF"):
                raise SpeechApiError(200, "not a WAV payload")
            check.duration_s = response.duration_s
            check.rtf = response.rtf
            check.best_cer = response.best_cer
            ft = client.asr(response.audio, "verify.wav", model="ft")
            base = client.asr(response.audio, "verify.wav", model="base")
            check.transcript_ft = ft.text
            check.transcript_base = base.text
            check.cer_ft = cer(text, ft.text)
            check.cer_base = cer(text, base.text)
        except SpeechApiError as exc:
            check.note = f"request failed: {exc}"
            report.checks.append(check)
            continue

        if voice in DIALECT_VOICES and check.cer_ft > max_cer:
            check.note = f"dialect ASR CER {check.cer_ft:.3f} exceeds {max_cer:.2f}"
        elif check.cer_base > max_cer:
            check.note = (
                f"stock Mandarin ASR CER {check.cer_base:.3f} exceeds {max_cer:.2f} "
                "(synthesis may be broken)"
            )
        check.ok = not check.note
        report.checks.append(check)

    if not report.checks:
        report.warnings.append("no checks ran")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sichuan-speech verify",
        description="End-to-end acceptance check against a running service.",
    )
    parser.add_argument("url", nargs="?", default="http://127.0.0.1:8100", help="service base URL")
    parser.add_argument("--n-samples", type=int, default=3, help="best-of-N candidates per request")
    parser.add_argument("--max-cer", type=float, default=0.25, help="per-voice CER ceiling")
    parser.add_argument("--expect-gpu", default="1", help="expected GPU id ('' to skip)")
    parser.add_argument("--json", dest="json_path", default="", help="also write the report as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = SpeechClient(args.url)
    try:
        report = verify(
            client,
            n_samples=args.n_samples,
            max_cer=args.max_cer,
            expect_gpu=args.expect_gpu,
        )
    except SpeechApiError as exc:
        print(f"cannot reach {args.url}: {exc}", file=sys.stderr)
        return 2
    print(report.summary())
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"wrote {args.json_path}")
    return 0 if report.ok else 1
