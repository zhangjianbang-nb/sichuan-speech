"""HTTP client: multipart encoding, header parsing, error surfacing, real round-trip."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sichuan_speech.client import (
    AsrResponse,
    SpeechApiError,
    SpeechClient,
    TtsResponse,
    _extract_detail,
    encode_multipart,
    quote_path,
)


def test_encode_multipart_has_boundary_and_parts():
    body, content_type = encode_multipart({"text": "你好", "voice": "sichuan"}, {})
    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=")[1].encode()
    assert boundary in body
    assert "你好".encode() in body
    assert b'name="text"' in body
    assert body.endswith(b"--" + boundary + b"--\r\n")


def test_encode_multipart_includes_file_and_sanitises_filename():
    body, content_type = encode_multipart(
        {"model": "ft"}, {"file": ('we"ird\nname.wav', b"RIFFdata", "audio/wav")}
    )
    assert content_type.startswith("multipart/form-data;")
    assert b'filename="we_ird_name.wav"' in body
    assert b"RIFFdata" in body
    assert b'name="model"' in body


def test_extract_detail_reads_fastapi_detail():
    assert _extract_detail(json.dumps({"detail": "text must not be empty"}).encode()) == "text must not be empty"


def test_extract_detail_serialises_list_detail():
    body = json.dumps({"detail": [{"loc": ["body", "text"], "msg": "required"}]}).encode()
    assert "loc" in (_extract_detail(body) or "")


def test_extract_detail_falls_back_to_plain_text():
    assert _extract_detail(b"upstream exploded") == "upstream exploded"
    assert _extract_detail(b"") is None


def test_tts_response_parses_headers():
    response = TtsResponse(
        audio=b"RIFF",
        headers={"x-duration": "1.25", "x-rtf": "0.31", "x-samples": "3", "x-best-cer": "0.0000", "x-sample-rate": "22050"},
    )
    assert response.duration_s == 1.25
    assert response.rtf == 0.31
    assert response.n_samples == 3
    assert response.best_cer == 0.0
    assert response.sample_rate == 22050


def test_tts_response_tolerates_missing_or_bad_headers():
    response = TtsResponse(audio=b"RIFF", headers={"x-duration": "abc", "x-samples": "many"})
    assert response.duration_s is None
    assert response.n_samples is None
    assert response.best_cer is None


def test_save_writes_bytes(tmp_path):
    response = TtsResponse(audio=b"RIFFabc", headers={})
    target = response.save(tmp_path / "nested" / "out.wav")
    assert target.read_bytes() == b"RIFFabc"


def test_quote_path_escapes_spaces():
    assert quote_path("a b.wav") == "a%20b.wav"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence test output
        return

    def _send(self, status: int, payload: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(200, json.dumps({"ok": True, "version": "2.0.0", "gpu": "1"}).encode(), {"Content-Type": "application/json"})
        elif self.path == "/voices":
            self._send(200, json.dumps({"voices": [{"id": "sichuan", "available": True}]}).encode())
        else:
            self._send(404, b'{"detail": "not found"}', {"Content-Type": "application/json"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if self.path == "/tts":
            if b"fail" in body:
                self._send(400, b'{"detail": "text must not be empty"}', {"Content-Type": "application/json"})
                return
            self._send(
                200,
                b"RIFF" + b"\x00" * 32,
                {
                    "Content-Type": "audio/wav",
                    "X-Duration": "1.00",
                    "X-RTF": "0.20",
                    "X-Samples": "2",
                    "X-Sample-Rate": "16000",
                },
            )
        elif self.path == "/asr":
            self._send(200, json.dumps({"text": "今天天气好安逸哦", "model": "ft", "elapsed_s": 0.5}).encode())
        else:
            self._send(404, b"nope")


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_client_round_trip(server):
    client = SpeechClient(server, timeout=5.0)
    assert client.health()["ok"] is True
    assert client.voices()["voices"][0]["id"] == "sichuan"

    response = client.tts("你好", n_samples=2)
    assert response.audio.startswith(b"RIFF")
    assert response.duration_s == 1.0
    assert response.n_samples == 2
    assert response.sample_rate == 16000

    asr = client.asr(b"RIFF", "a.wav", model="ft")
    assert isinstance(asr, AsrResponse)
    assert asr.text == "今天天气好安逸哦"
    assert asr.elapsed_s == 0.5


def test_client_surfaces_server_detail(server):
    client = SpeechClient(server, timeout=5.0)
    with pytest.raises(SpeechApiError) as excinfo:
        client.tts("fail")
    assert excinfo.value.status == 400
    assert "text must not be empty" in str(excinfo.value)


def test_client_rejects_non_wav_payload(monkeypatch: pytest.MonkeyPatch, server):
    client = SpeechClient(server, timeout=5.0)
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: (b"<html>", {}))
    with pytest.raises(SpeechApiError, match="non-WAV"):
        client.tts("你好")


def test_client_reports_unreachable_host():
    client = SpeechClient("http://127.0.0.1:9", timeout=0.5)
    with pytest.raises(SpeechApiError) as excinfo:
        client.health()
    assert excinfo.value.status == 0
    assert "cannot reach" in str(excinfo.value)
