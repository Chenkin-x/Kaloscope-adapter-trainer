"""The model-specific boundary: codecs, native time, forward and sampling schedule."""

import math
import sys
from pathlib import Path
import torch
from . import krea2, moss


def setup(cfg):
    if cfg["name"] == "krea2" and cfg.get("source"):
        source = Path(cfg["source"])
        if (source / "src").is_dir():
            source = source / "src"
        sys.path.insert(0, str(source))
    torch.backends.cuda.enable_cudnn_sdp(False)


def load_base(cfg, device):
    setup(cfg)
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    if cfg["name"] == "krea2":
        from musubi_tuner.krea2.krea2_utils import load_krea2_dit

        base = load_krea2_dit(cfg["dit"], device=device, dtype=dtype, attn_mode="torch")
    else:
        from .._vendor.moss import MOSSImageTransformer

        base = MOSSImageTransformer.from_pretrained(
            Path(cfg["model"]) / "transformer", torch_dtype=dtype, local_files_only=True
        ).to(device)
    return base.eval().requires_grad_(False)


def make_runner(base, cfg):
    return (krea2.Runner if cfg["backend"]["name"] == "krea2" else moss.Runner)(base, cfg["adapter"])


def native_prediction(base, name, z, time, text):
    if name == "krea2":
        return krea2.native_prediction(base, z, time, text)
    return base(z, time, text["text"], text["mask"]).sample


def model_time(name, sigma):
    return sigma if name == "krea2" else sigma * 1000


def training_sigma(name, values, latent_shape, shift=None):
    if shift is None:
        h, w = latent_shape[-2:]
        shift = (
            math.exp(0.5 + (math.ceil(h / 2) * math.ceil(w / 2) - 256) * 0.65 / 6144)
            if name == "krea2"
            else 1.0
        )
    if shift <= 0:
        raise ValueError("training_shift must be positive")
    return (shift * values / (1 + (shift - 1) * values)).clamp(1e-5, 1 - 1e-5)


def pack(z):
    b, c, h, w = z.shape
    return z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpack(z):
    b, c, h, w = z.shape
    return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


def load_vae(cfg, device):
    setup(cfg)
    if cfg["name"] == "krea2":
        from musubi_tuner.qwen_image.qwen_image_utils import load_vae as load

        return (
            load(cfg["vae"], device="cpu")
            .to(device=device, dtype=torch.bfloat16)
            .eval()
            .requires_grad_(False)
        )
    from diffusers import AutoencoderKLFlux2

    return (
        AutoencoderKLFlux2.from_pretrained(
            Path(cfg["model"]) / "vae", torch_dtype=torch.float32, local_files_only=True
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )


@torch.no_grad()
def encode_image(vae, name, pixels):
    device = next(vae.parameters()).device
    x = pixels.to(device).float() / 127.5 - 1
    if name == "krea2":
        return vae.encode_pixels_to_latents(x.to(torch.bfloat16))[:, :, 0]
    z = pack(vae.encode(x).latent_dist.mode())
    return (z - vae.bn.running_mean[None, :, None, None]) / (
        vae.bn.running_var[None, :, None, None] + vae.config.batch_norm_eps
    ).sqrt()


@torch.no_grad()
def decode_image(vae, name, z):
    if name == "krea2":
        return vae.decode_to_pixels(z.to(torch.bfloat16).unsqueeze(2)).float().clamp(0, 1)
    z = (
        z.float() * (vae.bn.running_var + vae.config.batch_norm_eps).sqrt()[None, :, None, None]
        + vae.bn.running_mean[None, :, None, None]
    )
    return (vae.decode(unpack(z), return_dict=False)[0].float().clamp(-1, 1) + 1) / 2


def load_text(cfg, device, max_length=128):
    setup(cfg)
    if cfg["name"] == "krea2":
        from musubi_tuner.krea2.krea2_utils import load_krea2_text_encoder

        return (
            load_krea2_text_encoder(
                cfg["text_encoder"],
                device=device,
                dtype=torch.bfloat16,
                max_length=max_length,
                tokenizer_repo=cfg["tokenizer"],
            )
            .eval()
            .requires_grad_(False)
        )
    from transformers import AutoModel, AutoTokenizer

    root = Path(cfg["model"])
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    model = (
        AutoModel.from_pretrained(root / "text_encoder", dtype=torch.bfloat16, local_files_only=True)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    return tokenizer, model, max_length


@torch.no_grad()
def encode_text(encoder, name, prompt):
    if name == "krea2":
        from musubi_tuner.krea2.krea2_sampling import gather_valid_text

        text, mask = encoder([prompt])
        text, mask = gather_valid_text(text, mask)
    else:
        tokenizer, model, length = encoder
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=length).to(
            next(model.parameters()).device
        )
        text = model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        mask = inputs["attention_mask"].bool()
    if not torch.isfinite(text).all() or not mask.any():
        raise ValueError("Invalid text encoding")
    return {"text": text, "mask": mask}


def sigmas(name, steps, start, shape, device):
    if steps < 1 or not 0 < start <= 1:
        raise ValueError("steps >= 1 and 0 < noise <= 1 required")
    if name == "moss":
        from .._vendor.moss import MOSSImageScheduler

        scheduler = MOSSImageScheduler(shift=5.0, training_shift=1.0)
        scheduler.set_timesteps(steps, device=device)
        return scheduler.sigmas.to(device)[scheduler.sigmas.to(device) <= start]
    h, w = shape[-2:]
    shift = math.exp(0.5 + (math.ceil(h / 2) * math.ceil(w / 2) - 256) * 0.65 / 6144)
    u = torch.linspace(start / (shift * (1 - start) + start), 0, steps + 1, device=device)
    sigma = shift * u / (1 + (shift - 1) * u)
    sigma[0] = start
    return sigma
