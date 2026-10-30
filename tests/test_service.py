"""Service behaviour with fake engines: best-of-N selection, validation, HTTP surface."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sichuan_speech.audio import load_audio
from sichuan_speech.config import Settings
from sichuan_speech.service import SpeechService, create_app


class FakeTts:
    def __init__(self, calls: list[tuple[str, str, float]], sample_rate: int = 16000):
        self.calls = calls
        self.sample_rate = sample_rate

    def synthesize(self, text: str, voice, *, speed: float = 1.0):
        self.calls.append((text, voice.id, speed))
        index = len(self.calls)
        wav = np.full(int(0.2 * self.sample_rate), index / 10, dtype=np.float32)
        return wav, self.sample_rate

    def list_speakers(self):
        return []


class FakeAsr:
    def __init__(self, text: str):
        self.text = text

    def transcribe(self, source) -> str:
        return self.text

    def transcribe_array(self, wav) -> str:
        return self.text


@pytest.fixture
def harness(ready_settings: Settings):
    calls: list[tuple[str, str, float]] = []
    scores = iter([0.5, 0.1, 0.0, 0.4, 0.2])
    service = SpeechService(
        ready_settings,
        tts_factory=lambda model_dir: FakeTts(calls),
        asr_factory=lambda finetuned: FakeAsr("方言回读" if finetuned else "普通话回读"),
        cer_fn=lambda text, wav, sr: next(scores),
    )
    app = create_app(ready_settings, service)
    return TestClient(app), calls


def test_health_and_voices(harness):
    client, _ = harness
    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["gpu"] == "1"
    assert health["voices"] >= 2
    assert health["loaded"] == {"sft": False, "base": False, "asr_ft": False, "asr_base": False}

    voices = client.get("/voices").json()
    ids = {voice["id"] for voice in voices["voices"]}
    assert {"sichuan", "mandarin"} <= ids
    assert voices["default"] == "sichuan"


def test_tts_picks_lowest_cer_and_reports_rendered_count(harness):
    client, calls = harness
    response = client.post(
        "/tts", data={"text": "今天天气好", "voice": "sichuan", "speed": "1.0", "n_samples": "3"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["x-samples"] == "3"
    assert float(response.headers["x-best-cer"]) == 0.0
    assert len(calls) == 3

    wav, sr = load_audio(response.content, target_sr=16000)
    assert sr == 16000
    # third candidate has amplitude 0.3 and the lowest CER, so it must be the one returned
    assert np.max(np.abs(wav)) == pytest.approx(0.3, abs=1e-3)


def test_tts_early_stops_at_zero_cer(ready_settings: Settings):
    calls: list[tuple[str, str, float]] = []
    scores = iter([0.0, 0.9])
    service = SpeechService(
        ready_settings,
        tts_factory=lambda model_dir: FakeTts(calls),
        asr_factory=lambda finetuned: FakeAsr("x"),
        cer_fn=lambda text, wav, sr: next(scores),
    )
    client = TestClient(create_app(ready_settings, service))
    response = client.post("/tts", data={"text": "你好", "voice": "sichuan", "n_samples": "5"})
    assert response.status_code == 200
    assert response.headers["x-samples"] == "1"
    assert len(calls) == 1


def test_single_sample_skips_scoring(harness):
    client, calls = harness
    response = client.post("/tts", data={"text": "你好", "voice": "sichuan", "n_samples": "1"})
    assert response.status_code == 200
    assert "x-best-cer" not in response.headers
    assert len(calls) == 1


def test_tts_validation_errors(harness):
    client, _ = harness
    assert client.post("/tts", data={"text": "   ", "voice": "sichuan"}).status_code == 400
    assert client.post("/tts", data={"text": "你好", "voice": "nope"}).status_code == 400
    assert (
        client.post("/tts", data={"text": "你好", "voice": "sichuan", "n_samples": "9"}).status_code
        == 400
    )
    assert (
        client.post("/tts", data={"text": "你好", "voice": "sichuan", "speed": "5"}).status_code
        == 400
    )


def test_tts_text_length_limit(harness):
    client, _ = harness
    response = client.post("/tts", data={"text": "字" * 501, "voice": "sichuan"})
    assert response.status_code == 400
    assert "too long" in response.json()["detail"]


def test_unavailable_voice_is_reported_not_crashed(settings: Settings):
    # No sft directory: the Sichuan voice must be listed as unavailable with a reason.
    service = SpeechService(
        settings, tts_factory=lambda model_dir: FakeTts([]), asr_factory=lambda ft: FakeAsr("x")
    )
    client = TestClient(create_app(settings, service))
    voices = {voice["id"]: voice for voice in client.get("/voices").json()["voices"]}
    assert voices["sichuan"]["available"] is False
    assert "missing" in voices["sichuan"]["reason"]

    response = client.post("/tts", data={"text": "你好", "voice": "sichuan"})
    assert response.status_code == 400
    assert "missing" in response.json()["detail"]


def test_asr_routes_to_selected_model(harness, tmp_path):
    client, _ = harness
    from sichuan_speech.audio import wav_bytes

    payload = wav_bytes(np.zeros(1600, dtype=np.float32), 16000)
    ft = client.post("/asr", files={"file": ("a.wav", payload, "audio/wav")}, data={"model": "ft"})
    base = client.post(
        "/asr", files={"file": ("a.wav", payload, "audio/wav")}, data={"model": "base"}
    )
    assert ft.status_code == 200
    assert ft.json()["text"] == "方言回读"
    assert base.json()["text"] == "普通话回读"
    assert ft.json()["model"] == "ft"


def test_asr_validation(harness):
    client, _ = harness
    assert client.post("/asr", files={"file": ("a.wav", b"", "audio/wav")}).status_code == 400
    assert (
        client.post(
            "/asr", files={"file": ("a.wav", b"RIFF", "audio/wav")}, data={"model": "wrong"}
        ).status_code
        == 400
    )


def test_asr_upload_limit(ready_settings: Settings):
    limited = ready_settings.with_overrides(max_upload_mb=1)
    service = SpeechService(
        limited, tts_factory=lambda d: FakeTts([]), asr_factory=lambda ft: FakeAsr("x")
    )
    client = TestClient(create_app(limited, service))
    big = b"0" * (1024 * 1024 + 10)
    response = client.post("/asr", files={"file": ("big.wav", big, "audio/wav")})
    assert response.status_code == 400
    assert "too large" in response.json()["detail"]


def test_api_index_exposes_limits(harness):
    client, _ = harness
    payload = client.get("/api").json()
    assert payload["endpoints"]["POST /tts"].startswith("form fields")
    assert payload["limits"]["max_text_chars"] == 500
    assert payload["limits"]["speed_range"] == [0.5, 2.0]


def test_web_ui_is_served(harness):
    client, _ = harness
    index = client.get("/")
    assert index.status_code == 200
    assert "四川话语音工坊" in index.text
    assert "app.js" in index.text
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
