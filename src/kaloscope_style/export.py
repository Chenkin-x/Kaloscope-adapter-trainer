"""Export canonical Krea2 Z3 checkpoints to the existing ComfyUI Z3 tensor format."""

import argparse
import json
from pathlib import Path
from safetensors.torch import save_file
from .io import tensor
from .adapter import StyleAdapter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("output")
    a = p.parse_args()
    cp = tensor(a.checkpoint)
    cfg = cp["config"]
    if cfg.get("backend", {}).get("name", "krea2") != "krea2":
        raise ValueError("The existing ComfyUI Z3 format supports Krea2 only")
    arch = dict(cfg.get("adapter", cfg))
    for key, value in dict(feature_dim=256, adapter_width=256, style_tokens=8, adapter_banks=4).items():
        if arch.get(key) != value:
            raise ValueError("The plugin supports the canonical Z3 architecture only")
    indices = arch.get("injection_indices", list(range(28)))
    slots = arch.get("injection_slots", [i // 7 for i in range(28)])
    if indices != list(range(28)) or slots != [i // 7 for i in range(28)]:
        raise ValueError("The plugin requires all 28 layers with four shared banks")
    arch.update(injection_indices=indices, injection_slots=slots, lora_rank=0)
    model = StyleAdapter()
    model.load_state_dict({k.removeprefix("adapter."): v for k, v in cp["trainable"].items()}, strict=True)
    import torch

    if not all(torch.isfinite(v).all() for v in model.parameters()):
        raise ValueError("Nonfinite weights")
    state = {"adapter." + k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    save_file(
        state,
        str(tmp),
        metadata={"format": "kaloscope_z3_v1", "config": json.dumps(arch), "step": str(cp["step"])},
    )
    tmp.replace(target)


if __name__ == "__main__":
    main()
