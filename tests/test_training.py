"""Exercise the public trainer, atomic checkpoint and deterministic resume on a tiny native MOSS."""

import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from kaloscope_style.data import cache_identity, cache_path
from kaloscope_style.io import atomic_json, save_tensor, tensor
from kaloscope_style.train import main


def fixture(tmp_path):
    pytest.importorskip("diffusers")
    from kaloscope_style._vendor.moss import MOSSImageTransformer

    torch.manual_seed(12)
    base = MOSSImageTransformer(
        dim=64, ffn_dim=96, num_heads=2, num_layers=4, text_dim=32, freq_dim=16, latent_channels=8
    )
    base.save_pretrained(tmp_path / "model/transformer")
    cfg = json.loads((Path(__file__).parents[1] / "configs/moss_z3.json").read_text())
    cfg.update(
        backend={"name": "moss", "model": str(tmp_path / "model")},
        manifest=str(tmp_path / "manifest.json"),
        encoder=str(tmp_path / "unused_encoder"),
        cache=str(tmp_path / "cache"),
        output=str(tmp_path / "run"),
    )
    cfg["adapter"] = dict(feature_dim=16, adapter_width=64, style_tokens=2, adapter_banks=2)
    cfg["training"].update(steps=4, global_batch=5, micro_batch=2, eval_every=2, save_every=2)
    doc = json.loads((Path(__file__).parents[1] / "examples/manifest.json").read_text())
    atomic_json(cfg["manifest"], doc)
    atomic_json(
        Path(cfg["cache"]) / "metadata.json",
        {"identity": cache_identity(cfg), "completed": ["features", "latents", "text"]},
    )
    for asset in doc["assets"]:
        save_tensor(
            cache_path(cfg["cache"], "features", asset["id"]),
            torch.nn.functional.normalize(torch.randn(1, 16), dim=1),
        )
        save_tensor(cache_path(cfg["cache"], "latents", asset["id"]), torch.randn(1, 8, 2, 2))
    for prompt in ["", doc["train"][0]["prompt"]]:
        save_tensor(
            cache_path(cfg["cache"], "text", prompt),
            dict(text=torch.randn(1, 3, 32), mask=torch.ones(1, 3, dtype=torch.bool)),
        )
    config_path = tmp_path / "config.json"
    atomic_json(config_path, cfg)
    return cfg, config_path


def test_public_trainer_resume(tmp_path, monkeypatch):
    cfg, path = fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--device", "cpu", "--stop-after", "2"])
    main()
    assert tensor(Path(cfg["output"]) / "latest.pt")["step"] == 2
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--device", "cpu", "--resume"])
    main()
    resumed = tensor(Path(cfg["output"]) / "latest.pt")
    assert resumed["step"] == 4
    cfg["output"] = str(tmp_path / "uninterrupted")
    atomic_json(path, cfg)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--device", "cpu"])
    main()
    full = tensor(Path(cfg["output"]) / "latest.pt")
    for key in full["trainable"]:
        torch.testing.assert_close(full["trainable"][key], resumed["trainable"][key], rtol=0, atol=0)
    logs = [json.loads(x) for x in (Path(cfg["output"]) / "steps.jsonl").read_text().splitlines()]
    assert logs[-1]["validation"]["n"] == 2


def test_uneven_ddp_matches_single_process(tmp_path, monkeypatch):
    cfg, path = fixture(tmp_path)
    cfg["training"].update(steps=1, warmup_steps=0, eval_every=0)
    atomic_json(path, cfg)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--device", "cpu"])
    main()
    single = tensor(Path(cfg["output"]) / "latest.pt")["trainable"]
    cfg["output"] = str(tmp_path / "distributed")
    atomic_json(path, cfg)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "kaloscope_style.train",
            "--config",
            str(path),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    multiple = tensor(Path(cfg["output"]) / "latest.pt")["trainable"]
    for key in single:
        torch.testing.assert_close(single[key], multiple[key], rtol=1e-4, atol=2e-7)


def test_all_reference_dropout_is_a_valid_control(tmp_path, monkeypatch):
    cfg, path = fixture(tmp_path)
    cfg["training"].update(steps=1, reference_dropout=1.0, eval_every=0)
    atomic_json(path, cfg)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--device", "cpu"])
    main()
    records = [json.loads(x) for x in (Path(cfg["output"]) / "steps.jsonl").read_text().splitlines()]
    assert records[0]["gradient_norm_rank0"] == 0.0
