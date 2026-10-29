"""Unified Sichuan speech API: TTS (CosyVoice2 SFT + zero-shot voices) and ASR (SenseVoice SFT).

CUDA_VISIBLE_DEVICES must be set before torch is imported, otherwise the process grabs
GPU0 and collides with ComfyUI. Keep that assignment as the very first statement.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("SICHUAN_GPU", "1")

import io, sys, time, tempfile
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

sys.path.append("/home/whu/CosyVoice")
sys.path.append("/home/whu/CosyVoice/third_party/Matcha-TTS")
sys.path.append("/home/whu/chuan/work")

SFT_DIR = os.path.expanduser(os.environ.get("SFT_DIR", "~/chuan/models/cosyvoice2-sichuan2"))
BASE_DIR = os.path.expanduser("~/models/CosyVoice2-0.5B")
ASR_CKPT = os.path.expanduser(os.environ.get("ASR_CKPT", "~/chuan/models/sensevoice-sichuan.pt"))
SICHUAN_PROMPT = os.path.expanduser("~/chuan/data/raw/magic_tts/1776909066881_g4plqjyx_32946.wav")
SICHUAN_PROMPT_TEXT = "领导想安排我转岗，我礼貌拒绝，我更想在现在的岗位发展。"
OFFICIAL_PROMPT = "/home/whu/CosyVoice/asset/zero_shot_prompt.wav"
OFFICIAL_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"

app = FastAPI(title="Sichuan speech API", version="2.0")
_state = {"base": None, "sft": None, "asr": None}


def get_sft():
    if _state["sft"] is None:
        from cosyvoice.cli.cosyvoice import AutoModel as CModel
        _state["sft"] = CModel(model_dir=SFT_DIR, fp16=True)
    return _state["sft"]


def get_base():
    if _state["base"] is None:
        from cosyvoice.cli.cosyvoice import AutoModel as CModel
        _state["base"] = CModel(model_dir=BASE_DIR, fp16=True)
    return _state["base"]


def get_asr():
    if _state["asr"] is None:
        import train_asr_lib as tal
        _state["asr"] = tal.load_finetuned(
            ASR_CKPT, device="cuda" if torch.cuda.is_available() else "cpu")
    return _state["asr"]


@app.get("/health")
def health():
    return {"ok": True, "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "sft_dir": SFT_DIR, "asr_ckpt": ASR_CKPT,
            "loaded": {k: v is not None for k, v in _state.items()}}


@app.get("/voices")
def voices():
    return {"voices": [
        {"id": "sichuan", "desc": "川话微调模型，成都女声，无需提示音频", "engine": "cosyvoice2-sft"},
        {"id": "sichuan_avg", "desc": "川话微调模型，2233 条川话说话人向量均值（平均音色）",
         "engine": "cosyvoice2-sft"},
        {"id": "sichuan_zeroshot", "desc": "原版模型 + 川话提示音零样本克隆（对照用，稳定性较差）",
         "engine": "cosyvoice2-zero-shot"},
        {"id": "mandarin", "desc": "原版模型普通话女声，原能力保留", "engine": "cosyvoice2-zero-shot"},
    ]}


@app.post("/tts")
def tts(text: str = Form(...), voice: str = Form("sichuan"), speed: float = Form(1.0)):
    text = text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    t0 = time.time()
    if voice in ("sichuan", "sichuan_avg"):
        m = get_sft()
        spk = "sichuan_female" if voice == "sichuan" else "sichuan_avg"
        if spk not in m.list_available_spks():
            raise HTTPException(400, "speaker %s not in this model" % spk)
        it = m.inference_sft(text, spk, stream=False, speed=speed)
    elif voice == "sichuan_zeroshot":
        m = get_base()
        it = m.inference_zero_shot(text, SICHUAN_PROMPT_TEXT, SICHUAN_PROMPT, stream=False, speed=speed)
    elif voice == "mandarin":
        m = get_base()
        it = m.inference_zero_shot(text, OFFICIAL_PROMPT_TEXT, OFFICIAL_PROMPT, stream=False, speed=speed)
    else:
        raise HTTPException(400, "unknown voice: %s" % voice)
    chunks = [o["tts_speech"].squeeze().cpu().numpy() for o in it]
    if not chunks:
        raise HTTPException(500, "no audio generated")
    wav = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    buf = io.BytesIO()
    sf.write(buf, wav, m.sample_rate, format="WAV")
    dur = len(wav) / m.sample_rate
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"X-Duration": "%.2f" % dur,
                             "X-RTF": "%.2f" % ((time.time() - t0) / max(1e-3, dur))})


@app.post("/asr")
def asr(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "a.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(file.file.read())
        p = f.name
    try:
        t0 = time.time()
        txt = get_asr().transcribe(p)
        return {"text": txt, "elapsed_s": round(time.time() - t0, 2)}
    finally:
        os.unlink(p)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8100")))
