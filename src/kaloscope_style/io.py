"""Explicit paths, atomic artifacts and content identities."""

import hashlib
import json
import os
from pathlib import Path
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def verify_provenance(metadata):
    """Fail if a model/source file changed in place after cache creation."""
    for path, expected in metadata.get("provenance", {}).items():
        if digest(path) != expected:
            raise ValueError(f"Cached model/source identity changed: {path}; rebuild in a new cache")


def read_config(path):
    # All configured paths resolve from the config's parent, never the caller's cwd.
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    for key in ("manifest", "cache", "output", "encoder", "initial_checkpoint"):
        if cfg.get(key):
            cfg[key] = str((path.parent / Path(cfg[key]).expanduser()).resolve())
    for key in ("model", "dit", "vae", "text_encoder", "tokenizer", "source"):
        if cfg["backend"].get(key):
            cfg["backend"][key] = str((path.parent / Path(cfg["backend"][key]).expanduser()).resolve())
    if cfg["backend"]["name"] not in ("krea2", "moss"):
        raise ValueError("backend.name must be krea2 or moss")
    for key in ("text_dropout", "reference_dropout"):
        if not 0 <= cfg["training"][key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if cfg["resolution"] < 32 or cfg["resolution"] % 16:
        raise ValueError("resolution must be a positive multiple of 16")
    for key in ("global_batch", "micro_batch", "steps", "save_every"):
        if cfg["training"][key] < 1:
            raise ValueError(f"{key} must be positive")
    training = cfg["training"]
    if training.get("sampling", "artist_balanced") not in ("artist_balanced", "shuffled_epochs"):
        raise ValueError("Unknown training.sampling")
    if not 0 < training["min_lr"] <= training["peak_lr"] or training["clip_grad"] <= 0:
        raise ValueError("Require 0 < min_lr <= peak_lr and positive clip_grad")
    return cfg


def load_adapter(runner, path):
    """Historical .pt trainable payload or adapter-only safetensors; no unsafe pickle."""
    if str(path).endswith(".safetensors"):
        from safetensors import safe_open
        from safetensors.torch import load_file

        state = load_file(str(path))
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            config = json.loads((handle.metadata() or {}).get("config", "{}"))
    else:
        cp = tensor(path)
        state = cp.get("trainable", cp.get("adapter", cp))
        config = cp.get("config", {})
    if not config:
        raise ValueError("Checkpoint must contain config metadata to validate injection placement")
    name = runner.__class__.__module__.rsplit(".", 1)[-1]
    if config.get("backend", {}).get("name", name) != name:
        raise ValueError("Checkpoint belongs to another image model")
    architecture = config.get("adapter", config)
    mapping = runner.slot if hasattr(runner, "slot") else runner.slots
    indices = architecture.get(
        "injection_indices",
        list(range(len(runner.base.blocks) if name == "krea2" else len(runner.base.layers))),
    )
    banks = architecture["adapter_banks"]
    slots = architecture.get("injection_slots", [i * banks // len(indices) for i in range(len(indices))])
    if len(slots) != len(indices) or dict(zip(indices, slots)) != mapping:
        raise ValueError("Checkpoint injection mapping differs from configured backend")
    if any(k.startswith("adapter.") for k in state):
        state = {k.removeprefix("adapter."): v for k, v in state.items()}
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite checkpoint weights")
    runner.adapter.load_state_dict(state, strict=True)
