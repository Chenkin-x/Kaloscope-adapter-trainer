"""Audited paired data and deterministic cached batches."""

import json
import random
from functools import lru_cache
from pathlib import Path
import torch
from .io import digest, fingerprint, tensor
from .backends.api import model_time, training_sigma


def manifest(path, check_files=False):
    path = Path(path)
    doc = json.loads(path.read_text())
    assets = {}
    for row in doc["assets"]:
        if row["id"] in assets or not row["artist"] or not row["family"]:
            raise ValueError("Asset IDs must be unique; artist/family must be nonempty")
        row = dict(row)
        row["path"] = str((path.parent / row["path"]).resolve())
        if row["split"] not in ("train", "dev", "test"):
            raise ValueError("Unknown split")
        if check_files and not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])
        assets[row["id"]] = row
    families, files = {}, {}
    for row in assets.values():
        # Family IDs are global, not artist-local, because duplicates can cross labels.
        for mapping, key in ((families, row["family"]), (files, row["path"])):
            if key in mapping and mapping[key] != row["split"]:
                raise ValueError("Work family / file leaks across train/dev/test")
            mapping[key] = row["split"]
    for split in ("train", "dev"):
        ids = set()
        for row in doc.get(split, []):
            if row["id"] in ids:
                raise ValueError("Duplicate pair ID")
            ids.add(row["id"])
            target = assets[row["target"]]
            if target["split"] != split or not row["references"]:
                raise ValueError("Target split mismatch or missing reference")
            for ref_id in row["references"]:
                ref = assets[ref_id]
                if (
                    ref["split"] != split
                    or ref["artist"] != target["artist"]
                    or ref["family"] == target["family"]
                ):
                    raise ValueError("References must be same artist, same split, different work family")
            if not isinstance(row["prompt"], str):
                raise ValueError("prompt must be text")
    if not doc.get("train"):
        raise ValueError("No training pairs")
    return doc, assets


def cache_identity(cfg):
    return fingerprint(
        {
            "backend": cfg["backend"],
            "encoder": cfg["encoder"],
            "resolution": cfg["resolution"],
            "encoder_resolution": cfg["encoder_resolution"],
            "text_max_length": cfg["text_max_length"],
            "manifest_sha256": digest(cfg["manifest"]),
        }
    )


def cache_path(root, kind, value):
    return Path(root) / kind / (fingerprint(value) + ".pt")


class CachedData:
    def __init__(self, cfg):
        self.cfg = cfg
        self.doc, self.assets = manifest(cfg["manifest"])
        metadata = json.loads((Path(cfg["cache"]) / "metadata.json").read_text())
        if metadata["identity"] != cache_identity(cfg) or set(metadata["completed"]) != {
            "features",
            "latents",
            "text",
        }:
            raise ValueError("Cache is incomplete or belongs to different models/data/config; rebuild it")
        self.metadata = metadata
        self.rows = self.doc["train"]
        self.by_artist = {}
        for row in self.rows:
            self.by_artist.setdefault(self.assets[row["target"]]["artist"], []).append(row)

    @lru_cache(maxsize=512)
    def read(self, kind, value):
        return tensor(cache_path(self.cfg["cache"], kind, value))

    def select(self, step, index):
        cfg = self.cfg["training"]
        rng = random.Random(cfg["seed"] + step * 65537 + index)
        if cfg.get("sampling", "artist_balanced") == "artist_balanced":
            return rng.choice(self.by_artist[rng.choice(sorted(self.by_artist))])
        # A deterministic bijection via shuffled epoch, cached once per epoch.
        offset = (step - 1) * cfg["global_batch"] + index
        epoch, pos = divmod(offset, len(self.rows))
        return self.rows[self.order(epoch)[pos]]

    @lru_cache(maxsize=2)
    def order(self, epoch):
        indices = list(range(len(self.rows)))
        random.Random(self.cfg["training"]["seed"] + epoch).shuffle(indices)
        return indices

    def batch(self, step, indices, device, rows=None, validation=False):
        examples = []
        for k, index in enumerate(indices):
            seed = self.cfg["training"]["seed"] + step * 65537 + index
            rng = random.Random(seed)
            row = rows[k] if rows is not None else self.select(step, index)
            ref = row["references"][0] if validation else rng.choice(row["references"])
            prompt = row["prompt"]
            if not validation and rng.random() < self.cfg["training"]["text_dropout"]:
                prompt = ""
            active = float(validation or rng.random() >= self.cfg["training"]["reference_dropout"])
            z = self.read("latents", row["target"]).float()
            gen = torch.Generator().manual_seed(seed)
            noise = torch.randn(z.shape, generator=gen)
            sigma = training_sigma(
                self.cfg["backend"]["name"],
                torch.randn(1, generator=gen).sigmoid(),
                z.shape,
                self.cfg["training"].get("time_shift"),
            )
            examples.append(
                (
                    z * (1 - sigma[:, None, None, None]) + noise * sigma[:, None, None, None],
                    sigma,
                    self.read("text", prompt),
                    self.read("features", ref),
                    active,
                    noise - z,
                )
            )
        length = max(e[2]["text"].shape[1] for e in examples)
        text = {}
        for key in ("text", "mask"):
            padded = []
            for e in examples:
                value = e[2][key]
                padded.append(
                    torch.cat((value, value.new_zeros((1, length - value.shape[1], *value.shape[2:]))), 1)
                )
            text[key] = torch.cat(padded).to(device)
        sigma = torch.cat([e[1] for e in examples]).to(device)
        return (
            torch.cat([e[0] for e in examples]).to(device),
            model_time(self.cfg["backend"]["name"], sigma),
            text,
            torch.cat([e[3] for e in examples]).to(device),
            torch.tensor([e[4] for e in examples], device=device),
            torch.cat([e[5] for e in examples]).to(device),
        )
