"""MOSS native block traversal with checkpoint-safe per-sample bindings."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from ..adapter import StyleAdapter


class Runner(nn.Module):
    def __init__(self, base, cfg):
        super().__init__()
        object.__setattr__(self, "base", base)
        self.adapter = StyleAdapter(
            cfg["feature_dim"],
            cfg["adapter_width"],
            cfg["style_tokens"],
            cfg["adapter_banks"],
            hidden=base.config.dim,
        ).to(next(base.parameters()).device)
        self.banks = cfg["adapter_banks"]
        indices = cfg.get("injection_indices", list(range(len(base.layers))))
        slots = cfg.get("injection_slots", [i * self.banks // len(indices) for i in range(len(indices))])
        if not indices or indices != sorted(set(indices)) or len(slots) != len(indices):
            raise ValueError("Invalid injection mapping")
        if any(i < 0 or i >= len(base.layers) for i in indices) or any(
            s < 0 or s >= self.banks for s in slots
        ):
            raise ValueError("Injection index/slot out of range")
        if set(slots) != set(range(self.banks)):
            raise ValueError("All configured adapter banks must be used")
        self.slots = dict(zip(indices, slots))
        self.use_checkpoint = True

    def forward(self, z, timestep, text, style, active, scale=1.0, skip_layers=()):
        m = self.base
        batch, _, height, width = z.shape
        n = height * width
        times = torch.as_tensor(timestep, device=z.device, dtype=torch.float32).reshape(-1).expand(batch)
        half = m.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=z.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = m.time_embedding(embedding.to(m.time_embedding[0].weight.dtype)).float()
        mod = m.shared_block_modulation(time.to(m.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = m.video_head_modulation(time.to(m.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        st = self.adapter.encode(style)
        outputs = []
        for index in range(batch):
            context = text["text"][index : index + 1, text["mask"][index].bool()]
            assert context.shape[1] > 0
            video = z[index : index + 1].flatten(2).transpose(1, 2)
            video = m.video_embedder(video.to(m.video_embedder.weight.dtype).contiguous())
            rope = m.rope(height, width, context.shape[1], z.device).to(video.dtype)
            context = m.text_embedder(context, rope[:, n:])
            x = torch.cat((video, context), 1)
            for layer_index, layer in enumerate(m.layers):
                if layer_index in skip_layers:
                    continue

                # Bind every sample-specific value: checkpoint recomputation must
                # never see a later sample's rope/modulation/activity mask.
                def fn(
                    v,
                    s,
                    layer=layer,
                    bank=self.slots.get(layer_index),
                    rope=rope,
                    modulation=mod[index : index + 1],
                    enabled=active[index : index + 1],
                ):
                    y = layer(v, rope, modulation)
                    return self.adapter(y, s, bank, n, enabled, scale) if bank is not None else y

                x = (
                    checkpoint(fn, x, st[index : index + 1], use_reentrant=False)
                    if self.training and self.use_checkpoint
                    else fn(x, st[index : index + 1])
                )
            x = m.final_norm_video(x[:, :n].to(m.final_norm_video.weight.dtype))
            x = x * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
            x = m.final_linear_video(x.to(m.final_linear_video.weight.dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        return torch.cat(outputs)
