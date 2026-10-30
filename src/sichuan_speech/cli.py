"""Command-line interface: ``sichuan-speech <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sichuan-speech",
        description="Sichuan dialect TTS/ASR toolkit: serve, verify, synthesise, transcribe.",
    )
    parser.add_argument("--version", action="version", version=f"sichuan-speech {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP service and web UI")
    serve.add_argument("--host", default=None, help="bind address (default from settings)")
    serve.add_argument("--port", type=int, default=None, help="port (default 8100)")
    serve.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value (default 1)")
    serve.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING")

    verify = sub.add_parser("verify", help="end-to-end acceptance check")
    verify.add_argument("url", nargs="?", default="http://127.0.0.1:8100")
    verify.add_argument("--n-samples", type=int, default=3)
    verify.add_argument("--max-cer", type=float, default=0.25)
    verify.add_argument("--expect-gpu", default="1")
    verify.add_argument("--json", dest="json_path", default="")

    batch = sub.add_parser("batch", help="generate a listenable demo batch")
    batch.add_argument("out_dir", nargs="?", default="demo/batch")
    batch.add_argument("--url", default="http://127.0.0.1:8100")
    batch.add_argument("--n-samples", type=int, default=3)

    variance = sub.add_parser("variance", help="measure sampling variance / best-of-N")
    variance.add_argument("n", nargs="?", type=int, default=6)
    variance.add_argument("--url", default="http://127.0.0.1:8100")
    variance.add_argument("--out", default="demo/variance")

    tts = sub.add_parser("tts", help="synthesise one sentence through a running service")
    tts.add_argument("text", help="text to speak")
    tts.add_argument("--voice", default="sichuan", help="voice id")
    tts.add_argument("--out", default="out.wav", help="output wav path")
    tts.add_argument("--speed", type=float, default=1.0)
    tts.add_argument("--n-samples", type=int, default=None)
    tts.add_argument("--url", default="http://127.0.0.1:8100")

    asr = sub.add_parser("asr", help="transcribe a file through a running service")
    asr.add_argument("file", help="audio file")
    asr.add_argument("--model", default="ft", choices=["ft", "base"])
    asr.add_argument("--url", default="http://127.0.0.1:8100")

    sub.add_parser("info", help="print resolved settings and which paths exist")
    voices = sub.add_parser("voices", help="list voices (local registry or a service)")
    voices.add_argument("--url", default="", help="query this service instead of local config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()

    if args.command == "serve":
        return _serve(args, settings)
    if args.command == "verify":
        from .verify import main as verify_main

        forwarded = [args.url]
        for flag, value in (
            ("--n-samples", args.n_samples),
            ("--max-cer", args.max_cer),
            ("--expect-gpu", args.expect_gpu),
        ):
            forwarded += [flag, str(value)]
        if args.json_path:
            forwarded += ["--json", args.json_path]
        return verify_main(forwarded)
    if args.command == "batch":
        from .batch import run_batch
        from .client import SpeechApiError, SpeechClient

        try:
            run_batch(SpeechClient(args.url), args.out_dir, n_samples=args.n_samples)
        except SpeechApiError as exc:
            print(f"failed: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.command == "variance":
        from .client import SpeechApiError, SpeechClient
        from .variance import run_probe

        try:
            run_probe(SpeechClient(args.url), args.n, args.out)
        except SpeechApiError as exc:
            print(f"failed: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.command == "tts":
        from .client import SpeechApiError, SpeechClient

        try:
            response = SpeechClient(args.url).tts(
                args.text, args.voice, speed=args.speed, n_samples=args.n_samples
            )
        except SpeechApiError as exc:
            print(f"failed: {exc}", file=sys.stderr)
            return 2
        path = response.save(args.out)
        print(
            f"wrote {path} ({response.duration_s or 0:.2f}s, rtf {response.rtf or 0:.2f}, "
            f"best CER {response.best_cer})"
        )
        return 0
    if args.command == "asr":
        from .client import SpeechApiError, SpeechClient

        try:
            result = SpeechClient(args.url).transcribe_file(args.file, model=args.model)
        except SpeechApiError as exc:
            print(f"failed: {exc}", file=sys.stderr)
            return 2
        print(result.text)
        return 0
    if args.command == "info":
        print(json.dumps(settings.describe(), ensure_ascii=False, indent=2))
        print("\nexisting paths:")
        for name, exists in settings.existing().items():
            print(f"  {'OK ' if exists else 'MISSING'} {name} = {getattr(settings, name)}")
        return 0
    if args.command == "voices":
        if args.url:
            from .client import SpeechApiError, SpeechClient

            try:
                payload = SpeechClient(args.url).voices()
            except SpeechApiError as exc:
                print(f"failed: {exc}", file=sys.stderr)
                return 2
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        from .service import SpeechService

        service = SpeechService(settings)
        print(
            json.dumps(
                {"default": "sichuan", "voices": service.voice_status()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return 2


def _serve(args: argparse.Namespace, settings: Settings) -> int:
    overrides = {}
    if args.host:
        overrides["host"] = args.host
    if args.port:
        overrides["port"] = args.port
    if args.gpu:
        overrides["gpu"] = args.gpu
    if args.log_level:
        overrides["log_level"] = args.log_level.upper()
    if overrides:
        settings = settings.with_overrides(**overrides)
    settings.configure_gpu()

    import uvicorn

    from .service import create_app

    app = create_app(settings)
    display_host = "127.0.0.1" if settings.host in {"0.0.0.0", "::"} else settings.host
    print(
        f"sichuan-speech {__version__} on {settings.host}:{settings.port} "
        f"(GPU {settings.gpu}, UI at http://{display_host}:{settings.port}/)"
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
