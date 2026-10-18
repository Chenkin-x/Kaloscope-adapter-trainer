import json
from pathlib import Path
import pytest
import torch
from kaloscope_style.adapter import StyleAdapter
from kaloscope_style.backends.api import pack, unpack, training_sigma, model_time
from kaloscope_style.data import manifest
from kaloscope_style.io import atomic_json
from kaloscope_style.metrics import paired_ci
from kaloscope_style.train import learning_rate


def test_adapter_identity_gradient_and_token_isolation():
    torch.manual_seed(3)
    adapter = StyleAdapter(16, 64, 2, 2, 64)
    x, style = torch.randn(2, 7, 64), torch.randn(2, 16)
    active = torch.tensor([1.0, 0.0])
    y = adapter(x, adapter.encode(style), 0, 4, active)
    torch.testing.assert_close(y, x, rtol=0, atol=0)
    y.square().mean().backward()
    assert adapter.out[0].weight.grad.norm() > 0
    torch.nn.init.normal_(adapter.out[0].weight, std=0.01)
    y = adapter(x, adapter.encode(style), 0, 4, active)
    assert not torch.equal(y[0, :4], x[0, :4])
    torch.testing.assert_close(y[:, 4:], x[:, 4:], rtol=0, atol=0)
    torch.testing.assert_close(y[1], x[1], rtol=0, atol=0)
    torch.testing.assert_close(adapter(x, adapter.encode(style), 0, 4, active, 0), x, rtol=0, atol=0)
    assert sum(p.numel() for p in StyleAdapter().parameters()) == 13308672
    assert sum(p.numel() for p in StyleAdapter(hidden=3072).parameters()) == 7017216


def test_vae_packing_and_time():
    z = torch.randn(2, 32, 6, 10)
    torch.testing.assert_close(unpack(pack(z)), z, rtol=0, atol=0)
    value = torch.tensor([0.5])
    torch.testing.assert_close(training_sigma("moss", value, z.shape), value)
    assert training_sigma("krea2", value, (1, 16, 64, 64)) > 0.5
    assert model_time("moss", value) == 500
    assert model_time("krea2", value) == 0.5


def test_work_family_leakage(tmp_path):
    doc = json.loads((Path(__file__).parents[1] / "examples/manifest.json").read_text())
    path = tmp_path / "manifest.json"
    atomic_json(path, doc)
    manifest(path)
    doc["assets"][4]["family"] = doc["assets"][0]["family"]
    atomic_json(path, doc)
    with pytest.raises(ValueError, match="leaks"):
        manifest(path)


def test_paired_ci_rejects_unpaired_and_clusters_artists():
    old = [dict(id=str(i), artist=str(i // 2), area=512, aspect="square", score=0.0) for i in range(8)]
    new = [{**r, "score": 0.1} for r in old]
    result = paired_ci(new, old, "score", repetitions=50)
    assert result["artists"] == 4 and result["mean"] == pytest.approx(0.1)
    assert result["low"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="Panel mismatch"):
        paired_ci(new[:-1], old, "score")


def test_lr_horizon():
    cfg = dict(steps=100, warmup_steps=10, peak_lr=1e-4, min_lr=1e-5)
    assert learning_rate(10, cfg) == pytest.approx(1e-4)
    assert learning_rate(100, cfg) == pytest.approx(1e-5)
