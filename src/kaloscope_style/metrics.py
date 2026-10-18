"""Independent CSD, symmetric views, and paired artist-cluster statistics."""

import importlib.util
import hashlib
import argparse
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F


def rgb(path):
    with Image.open(path) as raw:
        im = ImageOps.exif_transpose(raw).convert("RGBA")
        bg = Image.new("RGBA", im.size, "white")
        bg.alpha_composite(im)
        return bg.convert("RGB")


def views(im, size):
    # Three overlapping native-aspect square views, no artificial letterbox.
    w, h = im.size
    n = min(w, h)
    return [
        im.crop(
            (round((w - n) * p), round((h - n) * p), round((w - n) * p) + n, round((h - n) * p) + n)
        ).resize((size, size), Image.Resampling.BICUBIC)
        for p in (0, 0.5, 1)
    ]


class CSD:
    def __init__(self, root, device="cuda"):
        source = Path(root) / "vendor/CSD/models/clip/model.py"
        with source.open("rb") as f:
            assert (
                hashlib.file_digest(f, "sha256").hexdigest()
                == "889337fd6ba71672bb26c9b8927dcf07fdb2dbb49b7d4cb94c10b00e0dfc386e"
            ), "CSD source integrity failed"
        with (Path(root) / "csd-vit-l.bin").open("rb") as f:
            assert (
                hashlib.file_digest(f, "sha256").hexdigest()
                == "40e92fad63a361b8136100cd234c42d401ef9b34ff1748234318929ebcc7e7a1"
            ), "CSD weight integrity failed"
        spec = importlib.util.spec_from_file_location("kaloscope_csd_clip", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with torch.serialization.safe_globals(
            [
                argparse.Namespace,
                (np.core.multiarray.scalar, "numpy.core.multiarray.scalar"),
                np.dtype,
                type(np.dtype("float64")),
            ]
        ):
            state = torch.load(Path(root) / "csd-vit-l.bin", map_location="cpu", weights_only=True, mmap=True)
        state = state.get("model_state_dict", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model = module.VisionTransformer(224, 14, 1024, 24, 16, 768)
        backbone = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
        backbone["proj"] = state["last_layer_style"]
        self.model.load_state_dict(backbone, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.device = device
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)[None, :, None, None]
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)[None, :, None, None]

    @torch.inference_mode()
    def encode(self, im):
        batch = (
            torch.stack([torch.from_numpy(np.array(v)).permute(2, 0, 1) for v in views(im, 224)])
            .to(self.device)
            .float()
            / 255
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
            feature = F.normalize(self.model((batch - self.mean) / self.std).float(), dim=1)
        assert torch.isfinite(feature).all()
        return F.normalize(feature.mean(0), dim=0), feature


def palette(im):
    pixels = np.asarray(im.resize((128, 128), Image.Resampling.BICUBIC)).reshape(-1, 3)
    hist = np.concatenate([np.histogram(pixels[:, c], bins=16, range=(0, 256))[0] for c in range(3)]).astype(
        float
    )
    return hist / hist.sum()


def macro(rows, key):
    groups = defaultdict(list)
    for r in rows:
        if key in r:
            groups[(r["area"], r["aspect"], r["artist"])].append(float(r[key]))
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def summary(rows, keys):
    result = {"macro": {k: macro(rows, k) for k in keys}, "n": len(rows)}
    for dimension in ("area", "aspect", "artist", "cohort"):
        result["by_" + dimension] = {
            str(v): {k: macro([r for r in rows if r[dimension] == v], k) for k in keys}
            for v in sorted({r[dimension] for r in rows})
        }
    return result


def paired_ci(current, anchor, key, lower_better=False, repetitions=2000):
    """Paired bootstrap resamples ARTISTS, never crops/seeds as independent units."""
    a = {r["id"]: r for r in anchor}
    b = {r["id"]: r for r in current}
    if len(a) != len(anchor) or len(b) != len(current):
        raise ValueError("Duplicate paired IDs")
    if set(a) != set(b):
        raise ValueError("Panel mismatch: do not pair different cases")
    cells = defaultdict(list)
    for ident, r in b.items():
        old = a[ident]
        if any(r[k] != old[k] for k in ("artist", "area", "aspect")):
            raise ValueError("Pair metadata mismatch")
        value = float(r[key]) - float(old[key])
        value = -value if lower_better else value
        if not math.isfinite(value):
            raise ValueError("Nonfinite metric")
        cells[(r["artist"], r["area"], r["aspect"])].append(value)
    groups = defaultdict(list)
    for (artist, _, _), values in cells.items():
        groups[artist].append(np.mean(values))
    values = np.array([np.mean(groups[k]) for k in sorted(groups)])
    if len(values) < 4:
        raise ValueError("Too few independent artist clusters")
    rng = np.random.default_rng(20261012)
    means = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(1)
    low, high = np.quantile(means, [0.025, 0.975])
    return dict(
        mean=float(values.mean()),
        low=float(low),
        high=float(high),
        artists=len(values),
        method="paired artist-cluster percentile bootstrap; descriptive, not sequentially corrected",
    )
