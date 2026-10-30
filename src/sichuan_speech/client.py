"""HTTP client for the Sichuan speech service.

Only the standard library is used, so the client works on machines that cannot
install torch (the UI's verification scripts, CI smoke tests, a laptop talking to
the GPU box). Every call raises :class:`SpeechApiError` on non-2xx responses.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SpeechApiError(RuntimeError):
    """Non-2xx response or transport failure."""

    def __init__(self, status: int, message: str, body: bytes = b"") -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.body = body


def encode_multipart(
    fields: Mapping[str, Any],
    files: Mapping[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    """Encode form fields and files as ``multipart/form-data``.

    ``files`` maps the field name to ``(filename, content, content_type)``.
    """
    boundary = uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
            ).encode()
        )
    for name, (filename, content, content_type) in files.items():
        safe_name = filename.replace('"', "_").replace("\r", "_").replace("\n", "_")
        chunks.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                f'filename="{safe_name}"\r\nContent-Type: {content_type}\r\n\r\n'
            ).encode()
        )
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


@dataclass
class TtsResponse:
    audio: bytes
    headers: dict[str, str]

    @property
    def duration_s(self) -> float | None:
        return _as_float(self.headers.get("x-duration"))

    @property
    def rtf(self) -> float | None:
        return _as_float(self.headers.get("x-rtf"))

    @property
    def n_samples(self) -> int | None:
        value = self.headers.get("x-samples")
        return int(value) if value and value.isdigit() else None

    @property
    def best_cer(self) -> float | None:
        return _as_float(self.headers.get("x-best-cer"))

    @property
    def sample_rate(self) -> int | None:
        value = self.headers.get("x-sample-rate")
        return int(value) if value and value.isdigit() else None

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.audio)
        return target


@dataclass
class AsrResponse:
    text: str
    model: str
    elapsed_s: float


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class SpeechClient:
    """Thin wrapper over ``/health``, ``/voices``, ``/tts`` and ``/asr``."""

    def __init__(self, base_url: str = "http://127.0.0.1:8100", *, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    # ------------------------------------------------------------- primitives

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers=dict(headers or {})
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return response.read(), {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as exc:
            body = exc.read()
            message = _extract_detail(body) or exc.reason
            raise SpeechApiError(exc.code, message, body) from exc
        except urllib.error.URLError as exc:
            raise SpeechApiError(0, f"cannot reach {self.base_url}: {exc.reason}") from exc

    def _get_json(self, path: str, *, timeout: float | None = None) -> Any:
        body, _ = self._request("GET", path, timeout=timeout)
        return json.loads(body.decode("utf-8"))

    # ------------------------------------------------------------------- API

    def health(self) -> dict[str, Any]:
        return self._get_json("/health", timeout=10.0)

    def voices(self) -> dict[str, Any]:
        return self._get_json("/voices", timeout=10.0)

    def tts(
        self,
        text: str,
        voice: str = "sichuan",
        *,
        speed: float = 1.0,
        n_samples: int | None = None,
        timeout: float | None = None,
    ) -> TtsResponse:
        fields: dict[str, Any] = {"text": text, "voice": voice, "speed": speed}
        if n_samples is not None:
            fields["n_samples"] = n_samples
        body, content_type = encode_multipart(fields, {})
        audio, headers = self._request(
            "POST", "/tts", data=body, headers={"Content-Type": content_type}, timeout=timeout
        )
        if not audio.startswith(b"RIFF"):
            raise SpeechApiError(200, "server returned a non-WAV payload", audio[:200])
        return TtsResponse(audio=audio, headers=headers)

    def asr(
        self,
        audio: bytes,
        filename: str = "audio.wav",
        *,
        model: str = "ft",
        timeout: float | None = None,
    ) -> AsrResponse:
        body, content_type = encode_multipart(
            {"model": model}, {"file": (filename, audio, "audio/wav")}
        )
        payload, _ = self._request(
            "POST", "/asr", data=body, headers={"Content-Type": content_type}, timeout=timeout
        )
        data = json.loads(payload.decode("utf-8"))
        return AsrResponse(
            text=str(data.get("text", "")),
            model=str(data.get("model", model)),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
        )

    def synthesize_to(self, text: str, path: str | Path, **kwargs: Any) -> TtsResponse:
        response = self.tts(text, **kwargs)
        response.save(path)
        return response

    def transcribe_file(self, path: str | Path, *, model: str = "ft") -> AsrResponse:
        return self.asr(Path(path).read_bytes(), Path(path).name, model=model)

    def wait_ready(self, timeout: float = 60.0, interval: float = 1.0) -> dict[str, Any]:
        """Poll ``/health`` until the service answers or ``timeout`` expires."""
        deadline = time.monotonic() + timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self.health()
            except SpeechApiError as exc:
                last = exc
                time.sleep(interval)
        raise SpeechApiError(0, f"service not ready after {timeout:.0f}s: {last}")


def _extract_detail(body: bytes) -> str | None:
    """FastAPI puts errors in ``{"detail": ...}``; fall back to raw text."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        text = body.decode("utf-8", "replace").strip()
        return text[:300] if text else None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        return detail
    if detail is not None:
        return json.dumps(detail, ensure_ascii=False)[:300]
    return None


def quote_path(path: str) -> str:
    """URL-quote a path segment (kept public for tests and tooling)."""
    return urllib.parse.quote(path)
