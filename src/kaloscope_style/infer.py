"""Reference-guided txt2img / img2img with separate text and style guidance."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .backends import api
from .encoder import ArtistEncoder, pixels
from .io import read_config, load_adapter, atomic_json, digest


def guided(runner, name, x, time, text, empty, style, text_cfg, style_guidance, scale, native=False):
    active = torch.ones(x.shape[0], device=x.device)
    # MOSS's negative branch retains its native skip22; conditional branches use all layers.
    kwargs = {"skip_layers": (22,)} if name == "moss" else {}
    u = runner(x, time, empty, style, active, scale=0.0, **kwargs).float()
    t = runner(x, time, text, style, active, scale=0.0).float()
    if native:
        return u + text_cfg * (t - u)
    s = runner(x, time, text, style, active, scale=scale).float()
    return u + text_cfg * (t - u) + style_guidance * (s - t)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--content", help="Omit for txt2img; center-cropped square for img2img")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--text-cfg", type=float)
    parser.add_argument("--style-guidance", type=float)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.85)
    parser.add_argument("--native", action="store_true", help="No-style control, using the same text CFG")
    args = parser.parse_args()
    cfg = read_config(args.config)
    name = cfg["backend"]["name"]
    device = torch.device("cuda")
    artist = ArtistEncoder(cfg["encoder"]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        style = artist(pixels(args.reference, cfg["encoder_resolution"], allow_upscale=True)[None].to(device))
    del artist
    text_encoder = api.load_text(cfg["backend"], device, cfg["text_max_length"])
    text = api.encode_text(text_encoder, name, args.prompt)
    empty = api.encode_text(text_encoder, name, "")
    del text_encoder
    vae = api.load_vae(cfg["backend"], device)
    size = cfg["resolution"]
    shape = (1, 16, size // 8, size // 8) if name == "krea2" else (1, 128, size // 16, size // 16)
    content = (
        api.encode_image(vae, name, pixels(args.content, size, allow_upscale=True)[None])
        if args.content
        else None
    )
    vae.to("cpu")
    torch.cuda.empty_cache()
    base = api.load_base(cfg["backend"], device)
    runner = api.make_runner(base, cfg).eval()
    load_adapter(runner, args.checkpoint)
    sampling = cfg["inference"]
    steps = args.steps if args.steps is not None else sampling["steps"]
    text_cfg = args.text_cfg if args.text_cfg is not None else sampling["text_cfg"]
    style_guidance = args.style_guidance if args.style_guidance is not None else sampling["style_guidance"]
    noise = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(args.seed)).to(device)
    schedule = api.sigmas(name, steps, args.noise if content is not None else 1.0, shape, device)
    if len(schedule) < 2:
        raise ValueError("Noise threshold leaves no sampling steps")
    x = noise if content is None else content.float() * (1 - schedule[0]) + noise * schedule[0]
    for a, b in zip(schedule[:-1], schedule[1:]):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            velocity = guided(
                runner,
                name,
                x,
                api.model_time(name, a[None]),
                text,
                empty,
                style,
                text_cfg,
                style_guidance,
                args.scale,
                args.native,
            )
        x = x + (b - a) * velocity
        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite sampling trajectory")
    del runner, base
    torch.cuda.empty_cache()
    vae.to(device)
    output = api.decode_image(vae, name, x)
    # Krea2's decoder may retain the singleton frame dimension.
    if output.ndim == 5:
        output = output[:, :, 0]
    array = (output[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    atomic_json(
        path.with_suffix(".json"),
        {
            "checkpoint_sha256": digest(args.checkpoint),
            "image_sha256": digest(path),
            "backend": name,
            "reference_sha256": digest(args.reference),
            "prompt": args.prompt,
            "seed": args.seed,
            "text_cfg": text_cfg,
            "style_guidance": style_guidance,
            "scale": args.scale,
            "native": args.native,
            "sigmas": schedule.cpu().tolist(),
            "resolution": size,
            "content_sha256": digest(args.content) if args.content else None,
        },
    )


if __name__ == "__main__":
    main()
