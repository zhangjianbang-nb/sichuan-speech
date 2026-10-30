# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-10-30

### Added

- `sichuan_speech` installable package: configuration, text/CER, audio I/O, ASR,
  TTS, service, HTTP client, acceptance verification, batch generation, variance
  probe, data preparation, training and evaluation modules.
- `sichuan-speech` console entry point with `serve`, `verify`, `batch`, `variance`,
  `tts`, `asr`, `info`, `voices` and `version` subcommands.
- Built-in web UI (`sichuan_speech/web/`) served at `/`, with TTS playground,
  microphone/file ASR, automatic round-trip verification and a waveform view.
- `docs/data-collection.md`: corpus sources, download commands, cleaning rules,
  manifest schema, recording guide for your own data.
- `docs/inference.md`: service deployment, API reference, client examples,
  frontend usage and troubleshooting.
- Unit and integration tests (`tests/`), ruff + mypy configuration, GitHub Actions
  CI matrix on Python 3.10–3.12, `Makefile`, pre-commit hooks and a wheel-content
  check.
- Environment-driven configuration (`SICHUAN_*` variables) with documented defaults
  matching the reference training machine.

### Changed

- All hard-coded absolute paths (`/home/whu/...`, `~/chuan/...`) were moved into
  `Settings`; nothing outside `config.py` knows the machine layout.
- Duplicated `norm()`/`cer()` implementations (six copies) were replaced by
  `sichuan_speech.text`.
- Training/evaluation scripts are now importable modules with `main(argv)`; the
  `scripts/` directory keeps thin wrappers with the old command lines.
- The service gained input validation, request locking, structured errors,
  `X-Voice`/`X-Sample-Rate` headers, voice availability reporting and lazy model
  loading that never blocks `/health`.
- README rewritten around the new layout, training/data-collection/inference docs.

### Fixed

- Service no longer crashes on `/tts` when a voice's prompt audio is missing; the
  voice is reported as unavailable and requests return 400 with a clear message.
- `/asr` accepts arbitrary audio containers (wav/flac/ogg/webm) by falling back to
  ffmpeg when libsndfile cannot decode the file.
- Temporary files are always cleaned up, including on decode failures.
- `use_itn` is consistently disabled on both ASR paths, matching the evaluation
  protocol that produced the reported CER numbers.

## [1.0.0] - 2026-10-29

Initial release: CosyVoice2-0.5B LLM SFT and SenseVoiceSmall fine-tuning for
Sichuan dialect, evaluation reports, demo samples, FastAPI service and docs.
