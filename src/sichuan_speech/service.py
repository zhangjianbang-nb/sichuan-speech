"""FastAPI application factory for the Sichuan speech service.

Design notes
------------

* Models load lazily on the first request, so ``/health`` answers immediately and
  a broken checkpoint never prevents the process from starting.
* Every inference call is serialised with a re-entrant lock. The models are not
  thread-safe and a 24 GB card cannot hold two concurrent best-of-N renders.
* ``/tts`` can render ``n_samples`` candidates and keep the one whose round-trip
  CER against the input text is lowest. Sampling is stochastic: first-sample
  accuracy for a tricky word like 「归一」 is only 25–37%, best-of-5 raises it to
  76–90% at N× latency.
"""

from __future__ import annotations

import importlib.resources
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .audio import TARGET_SR, AudioDecodeError, duration, load_audio, wav_bytes
from .config import Settings
from .logging_utils import configure_logging
from .text import cer as compute_cer
from .tts import Voice, VoiceUnavailable, build_voices

logger = logging.getLogger(__name__)


class TtsEngine(Protocol):
    def synthesize(
        self, text: str, voice: Voice, *, speed: float = 1.0
    ) -> tuple[np.ndarray, int]: ...

    def list_speakers(self) -> list[str]: ...


class AsrEngine(Protocol):
    def transcribe_array(self, wav: np.ndarray) -> str: ...


@dataclass
class TtsResult:
    audio: bytes
    sample_rate: int
    duration_s: float
    rtf: float
    n_samples: int
    best_cer: float | None
    voice: str


@dataclass
class AsrResult:
    text: str
    model: str
    elapsed_s: float


class SpeechService:
    """Owns the model registry, validation and best-of-N selection."""

    def __init__(
        self,
        settings: Settings,
        *,
        tts_factory: Callable[[Path], TtsEngine] | None = None,
        asr_factory: Callable[[bool], AsrEngine] | None = None,
        cer_fn: Callable[[str, np.ndarray, int], float] | None = None,
    ) -> None:
        self.settings = settings
        self.voices = build_voices(settings)
        self._tts_factory = tts_factory or self._default_tts_factory
        self._asr_factory = asr_factory or self._default_asr_factory
        self._cer_fn = cer_fn
        self._lock = threading.RLock()
        self._engines: dict[str, Any] = {}

    # ------------------------------------------------------------------ engines

    def _default_tts_factory(self, model_dir: Path) -> TtsEngine:
        from .tts import CosyVoiceEngine

        return CosyVoiceEngine(model_dir, fp16=self.settings.tts_fp16, settings=self.settings)

    def _default_asr_factory(self, finetuned: bool) -> AsrEngine:
        from .asr import load_recognizer

        return load_recognizer(self.settings, finetuned=finetuned)

    def _engine(self, name: str) -> Any:
        with self._lock:
            if name not in self._engines:
                engine: Any
                if name == "sft":
                    engine = self._tts_factory(self.settings.sft_model_dir)
                elif name == "base":
                    engine = self._tts_factory(self.settings.base_model_dir)
                elif name == "asr_ft":
                    engine = self._asr_factory(True)
                elif name == "asr_base":
                    engine = self._asr_factory(False)
                else:  # pragma: no cover - internal typo guard
                    raise KeyError(name)
                self._engines[name] = engine
                logger.info("engine %s loaded", name)
            return self._engines[name]

    def loaded(self) -> dict[str, bool]:
        return {name: name in self._engines for name in ("sft", "base", "asr_ft", "asr_base")}

    # -------------------------------------------------------------- validation

    def validate_text(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("text must not be empty")
        if len(cleaned) > self.settings.max_text_chars:
            raise ValueError(
                f"text is too long: {len(cleaned)} characters "
                f"(limit {self.settings.max_text_chars})"
            )
        return cleaned

    def validate_voice(self, voice_id: str) -> Voice:
        voice = self.voices.get(voice_id)
        if voice is None:
            raise ValueError(f"unknown voice {voice_id!r}; available: {', '.join(self.voices)}")
        return voice

    def validate_speed(self, speed: float) -> float:
        if not self.settings.speed_min <= speed <= self.settings.speed_max:
            raise ValueError(
                f"speed must be within [{self.settings.speed_min}, {self.settings.speed_max}]"
            )
        return speed

    def validate_n_samples(self, n_samples: int) -> int:
        if not 1 <= n_samples <= self.settings.max_n_samples:
            raise ValueError(f"n_samples must be within [1, {self.settings.max_n_samples}]")
        return n_samples

    # ------------------------------------------------------------- availability

    def voice_status(self) -> list[dict[str, Any]]:
        status = []
        for voice in self.voices.values():
            available, reason = self._voice_available(voice)
            status.append(
                {
                    "id": voice.id,
                    "label": voice.label,
                    "description": voice.description,
                    "engine": voice.engine,
                    "available": available,
                    "reason": reason,
                }
            )
        return status

    def _voice_available(self, voice: Voice) -> tuple[bool, str]:
        if voice.engine == "sft":
            if not self.settings.sft_model_dir.exists():
                return False, f"model directory missing: {self.settings.sft_model_dir}"
            return True, ""
        if voice.prompt_wav is None or not Path(voice.prompt_wav).exists():
            return False, f"prompt audio missing: {voice.prompt_wav}"
        if not self.settings.base_model_dir.exists():
            return False, f"model directory missing: {self.settings.base_model_dir}"
        return True, ""

    # ------------------------------------------------------------------- health

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "version": __version__,
            "gpu": self.settings.gpu,
            "sft_dir": str(self.settings.sft_model_dir),
            "asr_ckpt": str(self.settings.asr_ckpt),
            "voices": len(self.voices),
            "loaded": self.loaded(),
        }

    # ---------------------------------------------------------------------- tts

    def self_cer(self, text: str, wav: np.ndarray, sample_rate: int) -> float:
        if self._cer_fn is not None:
            return float(self._cer_fn(text, wav, sample_rate))
        recognizer = self._engine("asr_ft")
        hypothesis = recognizer.transcribe_array(wav)
        return compute_cer(text, hypothesis)

    def synthesize(
        self,
        text: str,
        voice_id: str = "sichuan",
        speed: float = 1.0,
        n_samples: int | None = None,
    ) -> TtsResult:
        cleaned = self.validate_text(text)
        voice = self.validate_voice(voice_id)
        speed = self.validate_speed(float(speed))
        count = self.validate_n_samples(
            self.settings.default_n_samples if n_samples is None else int(n_samples)
        )
        available, reason = self._voice_available(voice)
        if not available:
            raise VoiceUnavailable(reason)

        with self._lock:
            engine = self._engine("sft" if voice.engine == "sft" else "base")
            started = time.perf_counter()
            best_wav: np.ndarray | None = None
            best_cer: float | None = None
            rendered = 0
            for _ in range(count):
                wav, sample_rate = engine.synthesize(cleaned, voice, speed=speed)
                rendered += 1
                if count == 1:
                    best_wav, best_cer = wav, None
                    break
                candidate_cer = self.self_cer(cleaned, wav, sample_rate)
                if best_wav is None or best_cer is None or candidate_cer < best_cer:
                    best_wav, best_cer = wav, candidate_cer
                if candidate_cer == 0.0:
                    break
            elapsed = time.perf_counter() - started
            assert best_wav is not None
            audio_duration = duration(best_wav, sample_rate)
            return TtsResult(
                audio=wav_bytes(best_wav, sample_rate),
                sample_rate=int(sample_rate),
                duration_s=audio_duration,
                rtf=elapsed / max(1e-3, audio_duration),
                n_samples=rendered,
                best_cer=best_cer,
                voice=voice.id,
            )

    # ---------------------------------------------------------------------- asr

    def transcribe(self, data: bytes, filename: str = "audio.wav", model: str = "ft") -> AsrResult:
        if model not in {"ft", "base"}:
            raise ValueError("model must be 'ft' (Sichuan fine-tune) or 'base' (stock Mandarin)")
        if not data:
            raise ValueError("uploaded file is empty")
        limit = self.settings.max_upload_mb * 1024 * 1024
        if len(data) > limit:
            raise ValueError(f"file is too large: {len(data)} bytes (limit {limit})")
        wav, _ = load_audio(data, target_sr=TARGET_SR)
        with self._lock:
            engine = self._engine("asr_ft" if model == "ft" else "asr_base")
            started = time.perf_counter()
            text = engine.transcribe_array(wav)
            elapsed = time.perf_counter() - started
        return AsrResult(text=text, model=model, elapsed_s=round(elapsed, 2))


def web_dir() -> Path:
    """Locate the bundled web UI (overridable with ``SICHUAN_WEB_DIR``)."""
    override = os.environ.get("SICHUAN_WEB_DIR")
    if override:
        return Path(override).expanduser().resolve()
    try:
        resource = importlib.resources.files("sichuan_speech").joinpath("web")
        path = Path(str(resource))
        if path.exists():
            return path
    except Exception:
        pass
    return Path(__file__).resolve().parent / "web"


def create_app(settings: Settings | None = None, service: SpeechService | None = None) -> FastAPI:
    """Build the FastAPI application."""
    settings = settings or Settings.from_env()
    service = service or SpeechService(settings)
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Sichuan speech API",
        description="Sichuan dialect TTS (CosyVoice2 SFT) and ASR (SenseVoice SFT).",
        version=__version__,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Duration", "X-RTF", "X-Samples", "X-Best-Cer", "X-Voice"],
        )

    app.state.settings = settings
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/voices")
    def voices() -> dict[str, Any]:
        return {"default": "sichuan", "voices": service.voice_status()}

    @app.get("/api")
    def api_index() -> dict[str, Any]:
        return {
            "name": "sichuan-speech",
            "version": __version__,
            "endpoints": {
                "GET /health": "liveness and loaded-model flags",
                "GET /voices": "voice registry with availability",
                "POST /tts": "form fields: text, voice, speed, n_samples -> audio/wav",
                "POST /asr": "multipart: file, model=ft|base -> JSON transcript",
                "GET /static/": "web UI assets",
                "GET /": "web UI",
            },
            "limits": {
                "max_text_chars": settings.max_text_chars,
                "max_upload_mb": settings.max_upload_mb,
                "max_n_samples": settings.max_n_samples,
                "speed_range": [settings.speed_min, settings.speed_max],
            },
            "config": settings.describe(),
        }

    @app.post("/tts")
    def tts(
        text: Annotated[str, Form()],
        voice: Annotated[str, Form()] = "sichuan",
        speed: Annotated[float, Form()] = 1.0,
        n_samples: Annotated[int | None, Form()] = None,
    ) -> Response:
        try:
            result = service.synthesize(text, voice, speed, n_samples)
        except (ValueError, VoiceUnavailable) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("TTS request failed")
            raise HTTPException(
                status_code=503, detail=f"TTS failed: {type(exc).__name__}: {exc}"
            ) from exc
        headers = {
            "X-Duration": f"{result.duration_s:.2f}",
            "X-RTF": f"{result.rtf:.2f}",
            "X-Samples": str(result.n_samples),
            "X-Voice": result.voice,
            "X-Sample-Rate": str(result.sample_rate),
        }
        if result.best_cer is not None:
            headers["X-Best-Cer"] = f"{result.best_cer:.4f}"
        return Response(result.audio, media_type="audio/wav", headers=headers)

    @app.post("/asr")
    def asr(
        file: Annotated[UploadFile, File()], model: Annotated[str, Form()] = "ft"
    ) -> dict[str, Any]:
        data = file.file.read()
        try:
            result = service.transcribe(data, file.filename or "audio.wav", model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AudioDecodeError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("ASR request failed")
            raise HTTPException(
                status_code=503, detail=f"ASR failed: {type(exc).__name__}: {exc}"
            ) from exc
        return {"text": result.text, "model": result.model, "elapsed_s": result.elapsed_s}

    assets = web_dir()
    if (assets / "index.html").exists():
        app.mount("/static", StaticFiles(directory=str(assets)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(assets / "index.html")

    return app


app = create_app()
