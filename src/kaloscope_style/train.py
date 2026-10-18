"""Portable frozen-backbone flow matching; single GPU or torchrun DDP."""

import argparse
import contextlib
import json
import math
import os
import signal
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .backends import api
from .data import CachedData
from .io import (
    read_config,
    save_tensor,
    atomic_json,
    tensor,
    digest,
    fingerprint,
    load_adapter,
    verify_provenance,
)

STOP = False


def stop(*_):
    global STOP
    STOP = True


def learning_rate(step, cfg):
    warmup = cfg.get("warmup_steps", 0)
    peak, floor = cfg["peak_lr"], cfg["min_lr"]
    if warmup and step <= warmup:
        return peak * step / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, cfg["steps"] - warmup)))
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def precision(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


def optimize(model, runner, optimizer, data, step, indices, global_batch, world, micro, device, lr, clip):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = lr
    total = 0.0
    active_count = 0
    for offset in range(0, len(indices), micro):
        selected = indices[offset : offset + micro]
        batch = data.batch(step, selected, device)
        active_count += int(batch[4].count_nonzero())
        # DDP averages gradients across ranks; compensate for uneven local counts.
        weight = len(selected) * world / global_batch
        sync = (
            model.no_sync()
            if isinstance(model, DDP) and offset + micro < len(indices)
            else contextlib.nullcontext()
        )
        with sync:
            with precision(device):
                prediction = model(*batch[:5])
            loss = (prediction.float() - batch[5].float()).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite FM loss")
            (loss * weight).backward()
            total += float(loss.detach()) * len(selected)
    grad = torch.nn.utils.clip_grad_norm_(runner.parameters(), clip)
    if not torch.isfinite(grad) or (grad <= 0 and active_count > 0):
        raise FloatingPointError("Missing/nonfinite adapter gradients")
    if any(p.grad is not None for p in runner.base.parameters()):
        raise RuntimeError("Frozen base unexpectedly received gradients")
    optimizer.step()
    return total, float(grad)


@torch.no_grad()
def validate(runner, data, device, rank=0, world=1):
    rows = data.doc.get("dev", [])
    if not rows:
        return None
    runner.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for i in range(rank, len(rows), world):
        batch = data.batch(900000, [i], device, rows=[rows[i]], validation=True)
        artist = data.assets[rows[i]["target"]]["artist"]
        wrong = next((r for r in rows if data.assets[r["target"]]["artist"] != artist), None)
        if wrong is None:
            raise ValueError("Fixed validation needs at least two dev artists for wrong-reference control")
        wrong_style = data.read("features", wrong["references"][0]).to(device)
        with precision(device):
            correct = runner(*batch[:5])
            off = runner(*batch[:5], scale=0.0)
            mismatched = runner(batch[0], batch[1], batch[2], wrong_style, batch[4])
        losses = [(p.float() - batch[5].float()).square().mean().item() for p in (correct, off, mismatched)]
        totals += torch.tensor([*losses, 1], device=device)
    if world > 1:
        dist.all_reduce(totals)
    runner.train()
    correct, off, wrong = (totals[:3] / totals[3]).tolist()
    return {
        "n": int(totals[3]),
        "correct_fm": correct,
        "off_fm": off,
        "wrong_fm": wrong,
        "condition_gain": (off - correct) / off,
        "correct_vs_wrong": (wrong - correct) / off,
        "protocol": "fixed validation seed 900000; same targets/noise/times; no dropout",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="Save and stop at this absolute step, preserving the full LR horizon",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    cfg = read_config(args.config)
    train = cfg["training"]
    rank, local, world = [
        int(os.environ.get(k, "0" if k != "WORLD_SIZE" else "1"))
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    ]
    if train["global_batch"] < world:
        raise ValueError("global_batch must be >= world_size")
    device = torch.device("cuda", local) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(4)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.manual_seed(train["seed"])
    data = CachedData(cfg)
    verify_provenance(data.metadata)
    base = api.load_base(cfg["backend"], device)
    runner = api.make_runner(base, cfg)
    optimizer = torch.optim.AdamW(
        runner.parameters(), lr=train["peak_lr"], weight_decay=train["weight_decay"]
    )
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "config": fingerprint(cfg),
        "cache": digest(Path(cfg["cache"]) / "metadata.json"),
        "world_size": world,
        "code": {
            str(p.relative_to(Path(__file__).parent)): digest(p) for p in Path(__file__).parent.rglob("*.py")
        },
    }
    first = 1
    if args.resume:
        cp = tensor(out / "latest.pt")
        if cp["identity"] != identity:
            raise ValueError("Resume refused: config/cache/code/world size changed; branch into a new run")
        runner.load_state_dict(cp["trainable"], strict=True)
        optimizer.load_state_dict(cp["optimizer"])
        torch.set_rng_state(cp["rng"][rank]["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(cp["rng"][rank]["cuda"], device)
        first = cp["step"] + 1
        if rank == 0 and (out / "steps.jsonl").exists():
            # A process may have logged an update after its last durable checkpoint.
            records = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
            records = [r for r in records if r["step"] < first]
            log_tmp = out / "steps.jsonl.tmp"
            log_tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
            log_tmp.replace(out / "steps.jsonl")
    else:
        if (out / "latest.pt").exists():
            raise FileExistsError("Run exists; use --resume or a new output directory")
        if cfg.get("initial_checkpoint"):
            load_adapter(runner, cfg["initial_checkpoint"])
    # Always test the disabled adapter, including nonzero historical initializations.
    batch = data.batch(0, [rank], device)
    runner.eval()
    with torch.no_grad(), precision(device):
        native = api.native_prediction(base, cfg["backend"]["name"], *batch[:3])
        disabled = runner(*batch[:5], scale=0.0)
    torch.testing.assert_close(disabled, native, rtol=0, atol=0)
    if rank == 0:
        atomic_json(
            out / "preflight.json",
            {
                "native_parity_max_abs": float((disabled - native).abs().max()),
                "trainable_parameters": sum(p.numel() for p in runner.parameters()),
                "world_size": world,
            },
        )
    del native, disabled, batch
    runner.train()
    model = (
        DDP(runner, device_ids=[local] if device.type == "cuda" else None, broadcast_buffers=False)
        if world > 1
        else runner
    )
    indices = list(range(rank, train["global_batch"], world))

    def save(step):
        rng = {"cpu": torch.get_rng_state()}
        if device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(device)
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng)
        else:
            states = [rng]
        if rank == 0:
            payload = {
                "trainable": runner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": states,
                "config": cfg,
                "step": step,
                "identity": identity,
            }
            save_tensor(out / "latest.pt", payload)
            save_tensor(
                out / f"adapter_step{step:08d}.pt",
                {"trainable": payload["trainable"], "config": cfg, "step": step},
            )
        if world > 1:
            dist.barrier()

    for step in range(first, train["steps"] + 1):
        tick = time.monotonic()
        loss, grad = optimize(
            model,
            runner,
            optimizer,
            data,
            step,
            indices,
            train["global_batch"],
            world,
            train["micro_batch"],
            device,
            learning_rate(step, train),
            train["clip_grad"],
        )
        total = torch.tensor(loss, device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(total)
        record = {
            "step": step,
            "flow_mse": float(total) / train["global_batch"],
            "gradient_norm_rank0": grad,
            "lr": learning_rate(step, train),
            "seconds": time.monotonic() - tick,
        }
        eval_every = train.get("eval_every", 0)
        if eval_every and step % eval_every == 0:
            save(step)  # Recoverable even if evaluation fails.
            record["validation"] = validate(runner, data, device, rank, world)
        if rank == 0:
            with (out / "steps.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)
        flag = torch.tensor(int(STOP or (args.stop_after and step >= args.stop_after)), device=device)
        if world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if step % train["save_every"] == 0 or step == train["steps"] or flag.item():
            save(step)
        if flag.item():
            break
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
