"""Real native tiny models; no weights, network or CUDA required."""

import pytest
import torch
from kaloscope_style.backends import moss, krea2


def verify(runner, native, z, t, text, style, active):
    torch.set_num_threads(2)
    runner.eval()
    expected = native(z, t, text)
    torch.testing.assert_close(runner(z, t, text, style, active), expected, rtol=0, atol=0)
    for layer in runner.adapter.out:
        torch.nn.init.normal_(layer.weight, std=0.01)
    torch.testing.assert_close(runner(z, t, text, style, active, scale=0.0), expected, rtol=0, atol=0)
    changed = dict(text=text["text"].clone(), mask=text["mask"])
    changed["text"][1, 2:] = 1000
    torch.testing.assert_close(
        runner(z, t, changed, style, active), runner(z, t, text, style, active), rtol=0, atol=0
    )
    runner.train()
    grads = []
    for enabled in (False, True):
        runner.zero_grad(set_to_none=True)
        runner.use_checkpoint = enabled
        runner(z, t, text, style, active).square().mean().backward()
        grads.append({k: p.grad.clone() for k, p in runner.named_parameters()})
    for key in grads[0]:
        torch.testing.assert_close(grads[0][key], grads[1][key], atol=1e-6, rtol=1e-5)
    assert all(p.grad is None for p in runner.base.parameters())
    assert all(not k.startswith("base.") for k in runner.state_dict())


def test_moss_native():
    pytest.importorskip("diffusers")
    from kaloscope_style._vendor.moss import MOSSImageTransformer

    torch.manual_seed(17)
    base = (
        MOSSImageTransformer(
            dim=64, ffn_dim=96, num_heads=2, num_layers=4, text_dim=32, freq_dim=16, latent_channels=8
        )
        .eval()
        .requires_grad_(False)
    )
    runner = moss.Runner(base, dict(feature_dim=16, adapter_width=64, style_tokens=2, adapter_banks=2))
    text = dict(text=torch.randn(2, 5, 32), mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]).bool())
    verify(
        runner,
        lambda z, t, txt: base(z, t, txt["text"], txt["mask"]).sample,
        torch.randn(2, 8, 2, 2),
        torch.tensor([100.0, 700.0]),
        text,
        torch.randn(2, 16),
        torch.tensor([1.0, 0.4]),
    )


def test_krea2_native():
    pytest.importorskip("musubi_tuner")
    from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT

    torch.manual_seed(17)
    cfg = SingleMMDiTConfig(
        features=128,
        tdim=16,
        txtdim=32,
        heads=2,
        kvheads=2,
        multiplier=2,
        layers=4,
        patch=2,
        channels=4,
        txtlayers=2,
        txtheads=2,
        txtkvheads=2,
    )
    base = SingleStreamDiT(cfg, attn_mode="torch").eval().requires_grad_(False)
    runner = krea2.Runner(base, dict(feature_dim=16, adapter_width=64, style_tokens=2, adapter_banks=2))
    text = dict(text=torch.randn(2, 5, 2, 32), mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]).bool())
    verify(
        runner,
        lambda z, t, txt: krea2.native_prediction(base, z, t, txt),
        torch.randn(2, 4, 4, 4),
        torch.tensor([0.1, 0.7]),
        text,
        torch.randn(2, 16),
        torch.tensor([1.0, 0.4]),
    )


def test_checkpoint_mapping_rejected(tmp_path):
    from kaloscope_style._vendor.moss import MOSSImageTransformer
    from kaloscope_style.io import save_tensor, load_adapter

    cfg = dict(feature_dim=16, adapter_width=64, style_tokens=2, adapter_banks=2)
    base = (
        MOSSImageTransformer(
            dim=64, ffn_dim=96, num_heads=2, num_layers=4, text_dim=32, freq_dim=16, latent_channels=8
        )
        .eval()
        .requires_grad_(False)
    )
    runner = moss.Runner(base, cfg)
    checkpoint = tmp_path / "adapter.pt"
    save_tensor(
        checkpoint, dict(trainable=runner.state_dict(), config={"backend": {"name": "moss"}, "adapter": cfg})
    )
    load_adapter(runner, checkpoint)
    cfg = {**cfg, "injection_slots": [1, 1, 0, 0]}
    save_tensor(
        checkpoint, dict(trainable=runner.state_dict(), config={"backend": {"name": "moss"}, "adapter": cfg})
    )
    with pytest.raises(ValueError, match="mapping"):
        load_adapter(runner, checkpoint)
