"""Score a fixed generated panel; references must be separate from conditioning works."""

import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .encoder import ArtistEncoder
from .metrics import CSD, rgb, views, summary, paired_ci
from .io import atomic_json
import numpy as np


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, help="JSON: cases + gallery; paths relative to panel")
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", help="Training-matched Kaloscope bundle (diagnostic)")
    parser.add_argument("--csd-root", help="Pinned independent CSD assets; see docs/EVALUATION.md")
    parser.add_argument("--anchor", help="Prior output JSON with exactly the same case IDs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.encoder and not args.csd_root:
        parser.error("Specify --encoder and/or --csd-root")
    path = Path(args.panel).resolve()
    panel = json.loads(path.read_text())
    cases, gallery = panel["cases"], panel["gallery"]
    if not cases or len({r["id"] for r in cases}) != len(cases) or len(gallery) < 2:
        raise ValueError("Unique case IDs and at least two gallery artists are required")
    # Prevent direct image reuse; work-family exclusion must also be audited upstream.
    score_paths = {str((path.parent / p).resolve()) for paths in gallery.values() for p in paths}
    for row in cases:
        if row["artist"] not in gallery:
            raise ValueError("Target artist absent from gallery")
        if str((path.parent / row["input_reference"]).resolve()) in score_paths:
            raise ValueError("Conditioning image leaked into scoring gallery")
    rows = [{k: r[k] for k in ("id", "artist", "area", "aspect", "cohort")} for r in cases]
    for name in ("kaloscope", "csd"):
        if name == "kaloscope" and not args.encoder or name == "csd" and not args.csd_root:
            continue
        model = (
            ArtistEncoder(args.encoder).to(args.device)
            if name == "kaloscope"
            else CSD(args.csd_root, args.device)
        )

        def encode(p, model=model, name=name):
            im = rgb(path.parent / p)
            if name == "csd":
                return model.encode(im)[0].cpu()
            batch = torch.stack([torch.from_numpy(np.array(v)).permute(2, 0, 1) for v in views(im, 512)]).to(
                args.device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                return F.normalize(model(batch).mean(0).float(), dim=0).cpu()

        artists = sorted(gallery)
        bank = torch.stack(
            [F.normalize(torch.stack([encode(p) for p in gallery[a]]).mean(0), dim=0) for a in artists]
        )
        for row, case in zip(rows, cases):
            sims = bank @ encode(case["image"])
            i = artists.index(case["artist"])
            row[name + "_similarity"] = float(sims[i])
            row[name + "_margin"] = float(sims[i] - (sims.sum() - sims[i]) / (len(sims) - 1))
            row[name + "_hit1"] = float(int(sims.argmax()) == i)
        del encode, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    keys = [k for k in rows[0] if k not in ("id", "artist", "area", "aspect", "cohort")]
    result = {
        "rows": rows,
        "summary": summary(rows, keys),
        "gallery_artists": len(gallery),
        "protocol": "three symmetric views; macro area/aspect/artist; gallery held apart from input",
    }
    if args.anchor:
        anchor = json.loads(Path(args.anchor).read_text())
        result["paired"] = {k: paired_ci(rows, anchor["rows"], k) for k in keys}
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
