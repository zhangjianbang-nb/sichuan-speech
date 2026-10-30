"""TTS-side training helpers: feature extraction, config derivation, model packaging.

CosyVoice2's SFT pipeline needs three pre-extracted artefacts per utterance
(campplus speaker embedding, speech tokens, parquet packs). ``prepare_features``
drives the checkout's own tools so the exact versions in use are the ones that
produce the features.

``build_inference_model`` turns a training checkpoint into an inference-ready
directory: the base model's files are symlinked, the checkpoint becomes
``llm.pt``, and the Sichuan speakers are written into ``spk2info.pt``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from ..config import Settings

logger = logging.getLogger(__name__)


def derive_sft_config(
    cosyvoice_home: str | Path,
    destination: str | Path,
    *,
    lr: float = 3e-6,
    max_epoch: int = 20,
    log_interval: int = 20,
) -> Path:
    """Patch the official ``cosyvoice2.yaml`` into an SFT config.

    Only three changes are made: speaker-conditioned padding, a low constant
    learning rate and denser logging. At lr 1e-5 the model overfits after one
    epoch on 6 hours of data; 3e-6 is what produced the usable 9-epoch curve.
    """
    source = Path(cosyvoice_home) / "examples" / "libritts" / "cosyvoice2" / "conf" / "cosyvoice2.yaml"
    if not source.exists():
        raise FileNotFoundError(f"CosyVoice config template not found: {source}")
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        "use_spk_embedding: False # change to True during sft", "use_spk_embedding: True"
    )
    text = text.replace("max_epoch: 200", f"max_epoch: {max_epoch}")
    text = text.replace("log_interval: 100", f"log_interval: {log_interval}")
    text = text.replace("lr: 1e-5 # change to 1e-5 during sft", f"lr: {lr} # Sichuan SFT")
    text = text.replace("scheduler: constantlr # change to constantlr during sft", "scheduler: constantlr")

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    logger.info("wrote %s (lr=%s, max_epoch=%d)", target, lr, max_epoch)
    return target


def prepare_features(
    settings: Settings,
    *,
    python: str | None = None,
    splits: Sequence[str] = ("train", "dev"),
    num_processes: int = 10,
    utts_per_parquet: int = 1000,
) -> dict:
    """Run CosyVoice's embedding/token/parquet tools for each split."""
    cosyvoice_home = settings.cosyvoice_home
    tools = cosyvoice_home / "tools"
    required = [
        tools / "extract_embedding.py",
        tools / "extract_speech_token.py",
        tools / "make_parquet_list.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "CosyVoice tools missing: " + ", ".join(missing) + ". Set SICHUAN_COSYVOICE_HOME."
        )
    interpreter = python or sys.executable
    onnx_dir = settings.base_model_dir
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(cosyvoice_home), str(cosyvoice_home / "third_party" / "Matcha-TTS"),
         env.get("PYTHONPATH", "")]
    )

    summary = {}
    for split in splits:
        directory = settings.data_root / "sft" / split
        if not (directory / "wav.scp").exists():
            raise FileNotFoundError(f"{directory}/wav.scp missing; build the manifest first")
        logger.info("== %s embedding", split)
        _run(
            [interpreter, str(tools / "extract_embedding.py"), "--dir", str(directory),
             "--onnx_path", str(onnx_dir / "campplus.onnx")],
            cwd=cosyvoice_home,
            env=env,
        )
        logger.info("== %s speech tokens", split)
        _run(
            [interpreter, str(tools / "extract_speech_token.py"), "--dir", str(directory),
             "--onnx_path", str(onnx_dir / "speech_tokenizer_v2.onnx")],
            cwd=cosyvoice_home,
            env=env,
        )
        parquet_dir = directory / "parquet"
        shutil.rmtree(parquet_dir, ignore_errors=True)
        parquet_dir.mkdir(parents=True, exist_ok=True)
        logger.info("== %s parquet", split)
        _run(
            [interpreter, str(tools / "make_parquet_list.py"),
             "--num_utts_per_parquet", str(utts_per_parquet),
             "--num_processes", str(num_processes),
             "--src_dir", str(directory), "--des_dir", str(parquet_dir)],
            cwd=cosyvoice_home,
            env=env,
        )
        data_list = parquet_dir / "data.list"
        lines = len(data_list.read_text(encoding="utf-8").splitlines()) if data_list.exists() else 0
        summary[split] = {"parquet_entries": lines}
    return summary


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    logger.debug("run: %s", " ".join(command))
    process = subprocess.run(command, cwd=str(cwd), env=env, capture_output=True, text=True)
    if process.returncode != 0:
        tail = (process.stderr or process.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}\n" + "\n".join(tail)
        )


def build_inference_model(
    settings: Settings,
    *,
    ckpt: str | Path,
    out: str | Path,
    prompt_wav: str | Path,
    prompt_text: str,
    spk_id: str = "sichuan_female",
    emb_pt: str | Path | None = None,
    utt2spk: str | Path | None = None,
    spk_prefix: str = "ks_",
    avg_spk_id: str = "sichuan_avg",
    base_dir: str | Path | None = None,
) -> dict:
    """Build an inference-ready CosyVoice2 directory from an SFT checkpoint."""
    import torch

    base = Path(base_dir) if base_dir else settings.base_model_dir
    if not base.exists():
        raise FileNotFoundError(f"base model directory not found: {base}")
    checkpoint = Path(ckpt)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)

    for name in os.listdir(base):
        if name in {"llm.pt", "spk2info.pt"}:
            continue
        destination = target / name
        if destination.exists() or destination.is_symlink():
            continue
        os.symlink(base / name, destination)

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    payload.pop("epoch", None)
    payload.pop("step", None)
    torch.save(payload, target / "llm.pt")
    logger.info("llm.pt written: %d keys", len(payload))

    from ..tts import _ensure_cosyvoice_importable

    _ensure_cosyvoice_importable(settings)
    from cosyvoice.cli.cosyvoice import AutoModel

    model = AutoModel(model_dir=str(target), fp16=True)
    model.add_zero_shot_spk(prompt_text, str(prompt_wav), spk_id)

    # add_zero_shot_spk stores llm_embedding/flow_embedding, but frontend_sft reads
    # spk2info[spk]['embedding'] -- two functions disagree about the schema.
    for value in model.frontend.spk2info.values():
        if isinstance(value, dict) and "embedding" not in value and "llm_embedding" in value:
            value["embedding"] = value["llm_embedding"]

    averaged = 0
    emb_path = Path(emb_pt) if emb_pt else settings.data_root / "sft" / "train" / "utt2embedding.pt"
    spk_path = Path(utt2spk) if utt2spk else settings.data_root / "sft" / "train" / "utt2spk"
    if emb_path.exists():
        embeddings = torch.load(str(emb_path), map_location="cpu", weights_only=False)
        keep: set[str] = set()
        if spk_path.exists():
            for line in spk_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[1].startswith(spk_prefix):
                    keep.add(parts[0])
        vectors = [
            (value if torch.is_tensor(value) else torch.tensor(value)).reshape(1, -1)
            for key, value in embeddings.items()
            if (not keep or key in keep)
        ]
        if vectors:
            mean = torch.cat(vectors, dim=0).mean(dim=0, keepdim=True)
            mean = torch.nn.functional.normalize(mean, dim=1)
            model.frontend.spk2info[avg_spk_id] = {"embedding": mean}
            averaged = len(vectors)
            logger.info("averaged embedding from %d utts", averaged)
        else:
            logger.warning("no utterances matched speaker prefix %r", spk_prefix)

    model.save_spkinfo()
    speakers = list(model.list_available_spks())
    logger.info("speakers: %s", speakers)
    summary = {
        "model_dir": str(target),
        "speakers": speakers,
        "llm_keys": len(payload),
        "averaged_utts": averaged,
    }
    (target / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return summary
