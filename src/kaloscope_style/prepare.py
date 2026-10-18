"""python -m kaloscope_style.prepare --config configs/krea2_z3.json --kind all"""

import argparse
from pathlib import Path
import torch
from . import encoder
from .backends import api
from .data import manifest, cache_identity, cache_path
from .io import read_config, atomic_json, save_tensor, digest, verify_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--kind", choices=["audit", "features", "latents", "text", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = read_config(args.config)
    doc, assets = manifest(cfg["manifest"], check_files=True)
    # Only train/dev images are read; sealed test files are never encoded.
    used = {
        i for split in ("train", "dev") for r in doc.get(split, []) for i in [r["target"], *r["references"]]
    }
    hashes = {i: digest(assets[i]["path"]) for i in sorted(used)}
    seen = {}
    for i, sha in hashes.items():
        split = assets[i]["split"]
        if sha in seen and seen[sha] != split:
            raise ValueError("Exact duplicate image crosses splits")
        seen[sha] = split
    audit = {
        "train_pairs": len(doc["train"]),
        "dev_pairs": len(doc.get("dev", [])),
        "image_sha256": hashes,
        "cross_family_verified": True,
    }
    if args.kind == "audit":
        print(audit)
        return
    root = Path(cfg["cache"])
    root.mkdir(parents=True, exist_ok=True)
    identity = cache_identity(cfg)
    metadata_path = root / "metadata.json"
    import json

    old = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if old and (old["identity"] != identity or old["audit"] != audit):
        raise ValueError("Cache identity/image contents changed: use a new cache directory")
    if old:
        verify_provenance(old)
    # Capture weight and source hashes once. Replacing weights requires a new cache.
    if not old:
        paths = [Path(cfg["encoder"])]
        paths += [
            Path(v)
            for k, v in cfg["backend"].items()
            if k in ("model", "dit", "vae", "text_encoder", "tokenizer", "source")
        ]
        provenance = {}
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(p)
            files = (
                [p]
                if p.is_file()
                else sorted(
                    x
                    for x in p.rglob("*")
                    if x.is_file() and x.suffix in (".safetensors", ".pt", ".bin", ".json", ".py")
                )
            )
            for f in files:
                provenance[str(f)] = digest(f)
        old = {"identity": identity, "completed": [], "audit": audit, "provenance": provenance}
        atomic_json(metadata_path, old)
    kinds = ["features", "latents", "text"] if args.kind == "all" else [args.kind]
    name = cfg["backend"]["name"]
    for kind in kinds:
        if kind == "features":
            model = encoder.ArtistEncoder(cfg["encoder"]).to(args.device)
            for i in sorted(used):
                dest = cache_path(root, kind, i)
                if dest.exists():
                    continue
                x = encoder.pixels(assets[i]["path"], cfg["encoder_resolution"])[None].to(args.device)
                with (
                    torch.no_grad(),
                    torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")),
                ):
                    result = model(x)
                if result.shape != (1, cfg["adapter"]["feature_dim"]) or not torch.isfinite(result).all():
                    raise ValueError("Wrong/nonfinite style features")
                save_tensor(dest, result.cpu())
        elif kind == "latents":
            model = api.load_vae(cfg["backend"], args.device)
            ids = sorted({r["target"] for s in ("train", "dev") for r in doc.get(s, [])})
            for i in ids:
                dest = cache_path(root, kind, i)
                if dest.exists():
                    continue
                result = api.encode_image(
                    model, name, encoder.pixels(assets[i]["path"], cfg["resolution"])[None]
                )
                if not torch.isfinite(result).all():
                    raise ValueError("Nonfinite latent")
                save_tensor(dest, result.cpu())
        else:
            model = api.load_text(cfg["backend"], args.device, cfg["text_max_length"])
            prompts = sorted({""} | {r["prompt"] for s in ("train", "dev") for r in doc.get(s, [])})
            for prompt in prompts:
                dest = cache_path(root, kind, prompt)
                if not dest.exists():
                    result = api.encode_text(model, name, prompt)
                    save_tensor(dest, {k: v.cpu() for k, v in result.items()})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        old["completed"] = sorted(set(old["completed"]) | {kind})
        atomic_json(metadata_path, old)
        print(f"{kind}: complete", flush=True)


if __name__ == "__main__":
    main()
