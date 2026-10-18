"""Training-matched artist projector; self-contained DINOv3 inference subset."""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from kaloscope_dinov3.architecture import build_backbone
from .io import tensor


def rgb(path):
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGBA")
        bg = Image.new("RGBA", image.size, "white")
        bg.alpha_composite(image)
        return bg.convert("RGB")


def pixels(path, size=512, allow_upscale=False):
    image = rgb(path)
    n = min(image.size)
    if n < size and not allow_upscale:
        raise ValueError(f"Upscale disabled: {path}, {image.size}, requested {size}")
    left, top = (image.width - n) // 2, (image.height - n) // 2
    image = image.crop((left, top, left + n, top + n)).resize((size, size), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)


class ArtistEncoder(nn.Module):
    def __init__(self, path):
        super().__init__()
        path = Path(path)
        if path.is_dir():
            config = json.loads((path / "config.json").read_text())
            from safetensors.torch import load_file

            state = load_file(str(path / config.get("checkpoint", "model.safetensors")))
            config = config["model"]
        else:
            cp = tensor(path)
            config, state = cp["model_config"], cp["model"]
        self.pooling = config.get("pooling", "cls_mean")
        self.backbone = build_backbone(config)
        self.backbone.load_state_dict(
            {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")},
            strict=True,
        )
        hidden, dim = state["projector.0.weight"].shape
        output = state["projector.2.weight"].shape[0]
        self.projector = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, output))
        self.projector.load_state_dict(
            {k.removeprefix("projector."): v for k, v in state.items() if k.startswith("projector.")},
            strict=True,
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, images):
        result = self.backbone.forward_features((images.float() / 255 - self.mean) / self.std)
        cls, patch = result["x_norm_clstoken"], result["x_norm_patchtokens"].mean(1)
        pooled = (
            cls if self.pooling == "cls" else patch if self.pooling == "mean" else torch.cat((cls, patch), 1)
        )
        return F.normalize(self.projector(pooled).float(), dim=1)
