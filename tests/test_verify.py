"""Acceptance check logic with a fake client (no service needed)."""

from __future__ import annotations

from sichuan_speech.client import AsrResponse, SpeechApiError, TtsResponse
from sichuan_speech.verify import CheckResult, _display_width, _fmt, _pad, verify


class FakeClient:
    def __init__(self, transcripts: dict[str, str] | None = None, fail_voice: str | None = None):
        self.transcripts = transcripts or {}
        self.fail_voice = fail_voice

    def health(self):
        return {"ok": True, "version": "2.0.0", "gpu": "1"}

    def voices(self):
        return {"default": "sichuan", "voices": [{"id": "sichuan"}, {"id": "mandarin"}]}

    def tts(self, text, voice="sichuan", *, speed=1.0, n_samples=None):
        if voice == self.fail_voice:
            raise SpeechApiError(503, "engine exploded")
        return TtsResponse(
            audio=b"RIFF" + b"\x00" * 32,
            headers={"x-duration": "1.50", "x-rtf": "0.25", "x-samples": str(n_samples or 1), "x-best-cer": "0.0"},
        )

    def asr(self, audio, filename="a.wav", *, model="ft"):
        key = f"{filename}:{model}"
        text = self.transcripts.get(key, self.transcripts.get(model, "今天天气好"))
        return AsrResponse(text=text, model=model, elapsed_s=0.2)


def test_verify_passes_when_transcripts_match():
    report = verify(FakeClient({"ft": "今天天气好", "base": "今天天气好"}), [("sichuan", "今天天气好")], n_samples=1)
    assert report.ok is True
    assert report.checks[0].cer_ft == 0.0
    assert report.checks[0].note == ""
    assert report.as_dict()["checks"][0]["voice"] == "sichuan"


def test_verify_flags_dialect_cer_above_threshold():
    report = verify(FakeClient({"ft": "今天下雨了", "base": "今天天气好"}), [("sichuan", "今天天气好")], max_cer=0.2)
    assert report.ok is False
    assert "dialect ASR CER" in report.checks[0].note


def test_verify_flags_broken_synthesis_for_mandarin_voice():
    report = verify(FakeClient({"ft": "乱码", "base": "乱码"}), [("mandarin", "今天天气好")], max_cer=0.2)
    assert report.ok is False
    assert "synthesis may be broken" in report.checks[0].note


def test_verify_records_request_failures():
    report = verify(FakeClient(fail_voice="sichuan"), [("sichuan", "你好")], n_samples=1)
    assert report.ok is False
    assert report.checks[0].note.startswith("request failed")
    assert report.fatal is True


def test_verify_warns_on_gpu_and_missing_voices():
    class ThinClient(FakeClient):
        def health(self):
            return {"ok": True, "gpu": "0"}

        def voices(self):
            return {"voices": [{"id": "sichuan"}]}

    report = verify(ThinClient({"ft": "你好", "base": "你好"}), [("sichuan", "你好")], expect_gpu="1")
    assert any("GPU" in warning for warning in report.warnings)
    assert any("mandarin" in warning for warning in report.warnings)
    assert report.ok is True  # warnings do not fail the run


def test_verify_without_checks_is_not_ok():
    report = verify(FakeClient(), [], expect_gpu="")
    assert report.ok is False
    assert "no checks ran" in report.warnings


def test_summary_and_dict_are_serialisable():
    report = verify(FakeClient({"ft": "今天天气好", "base": "今天天气好"}), [("sichuan", "今天天气好")], n_samples=2)
    payload = report.as_dict()
    assert payload["ok"] is True
    assert payload["checks"][0]["duration_s"] == 1.5
    assert isinstance(report.summary(), str)
    assert "sichuan" in report.summary()


def test_check_result_defaults_are_failing():
    check = CheckResult(voice="sichuan", text="你好")
    assert check.ok is False
    assert check.cer_ft == 1.0


def test_table_helpers_align_cjk():
    assert _display_width("abc") == 3
    assert _display_width("中文") == 4
    assert _pad("中文", 6) == "中文  "
    assert _fmt(None) == "-"
    assert _fmt(0.12345) == "0.1235"
