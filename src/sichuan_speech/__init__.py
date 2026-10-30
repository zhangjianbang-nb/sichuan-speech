"""Sichuan dialect speech toolkit: CosyVoice2 TTS + SenseVoice ASR.

The package is deliberately split so that the heavy stacks (torch, funasr,
cosyvoice) are only imported when a model is actually loaded:

* :mod:`sichuan_speech.config`  -- environment-driven paths and limits
* :mod:`sichuan_speech.text`    -- transcript normalisation and CER
* :mod:`sichuan_speech.audio`   -- decoding, resampling and WAV encoding
* :mod:`sichuan_speech.asr`     -- SenseVoice recognisers
* :mod:`sichuan_speech.tts`     -- CosyVoice2 engines and the voice registry
* :mod:`sichuan_speech.service` -- FastAPI application factory
* :mod:`sichuan_speech.client`  -- HTTP client used by the CLIs and the UI
"""

from __future__ import annotations

__version__ = "2.0.0"
__all__ = ["__version__"]
