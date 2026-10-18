"""Krea2 native block traversal, adapted from the verified AI0/Z3 runner."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from ..adapter import StyleAdapter


class Runner(nn.Module):
    def __init__(self, base, cfg):
        super().__init__()
        object.__setattr__(self, "base", base)
        self.cfg = cfg
        self.adapter = StyleAdapter(
            cfg["feature_dim"],
            cfg["adapter_width"],
            cfg["style_tokens"],
            cfg["adapter_banks"],
            hidden=base.config.features,
        ).to(next(base.parameters()).device)
        # The public Z3 package trains only the sidecar, never base LoRA.
        self.use_checkpoint = True
        self.indices = cfg.get("injection_indices", list(range(len(base.blocks))))
        assert self.indices and self.indices == sorted(set(self.indices))
        assert all(0 <= i < len(base.blocks) for i in self.indices)
        slots = cfg.get(
            "injection_slots",
            [i * cfg["adapter_banks"] // len(self.indices) for i in range(len(self.indices))],
        )
        if len(slots) != len(self.indices) or set(slots) != set(range(cfg["adapter_banks"])):
            raise ValueError("Each injection needs one valid bank; all configured banks must be used")
        self.slot = dict(zip(self.indices, slots))
        assert len(self.slot) == len(self.indices) and all(
            0 <= i < len(self.adapter.out) for i in self.slot.values()
        )

    def forward(self, z, t, text, style, active, scale=1.0):
        from musubi_tuner.krea2.krea2_sampling import prepare
        from musubi_tuner.krea2.krea2_mmdit import temb, AttentionParams

        m = self.base
        img, pos, mask = prepare(z, text["text"].shape[1], 2, text["mask"].bool())
        n = img.shape[1]
        img = m.first(img)
        tv = m.tmlp(temb(t, m.config.tdim, device=img.device, dtype=img.dtype))
        tvec = m.tproj(tv)
        txtmask = mask[:, n:]
        no = AttentionParams.create_attention_params_from_mask(m.attn_mode, m.split_attn, 0, None)
        yes = AttentionParams.create_attention_params_from_mask(m.attn_mode, m.split_attn, 0, txtmask)
        context = m.txtmlp(m.txtfusion(text["text"], no, yes))
        x = torch.cat((img, context), dim=1)
        pad = (-x.shape[1]) % 256
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            pos = F.pad(pos, (0, 0, 0, pad))
            txtmask = F.pad(txtmask, (0, pad), value=False)
        ap = AttentionParams.create_attention_params_from_mask(m.attn_mode, m.split_attn, n, txtmask)
        freq = m.posemb(pos)
        st = self.adapter.encode(style)
        for i, block in enumerate(m.blocks):
            # Frozen downstream blocks still propagate gradients to an earlier injection.
            if i in self.slot:

                def fn(v, s, block=block, index=self.slot[i]):
                    return self.adapter(block(v, tvec, freq, ap), s, index, n, active, scale)

                x = (
                    checkpoint(fn, x, st, use_reentrant=False)
                    if self.training and self.use_checkpoint
                    else fn(x, st)
                )
            else:

                def fn(v, block=block):
                    return block(v, tvec, freq, ap)

                needs_graph = i > self.indices[0]
                x = (
                    checkpoint(fn, x, use_reentrant=False)
                    if self.training and self.use_checkpoint and needs_graph
                    else fn(x)
                )
        y = m.last(x, tv)[:, :n]
        return rearrange(
            y, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=z.shape[-2] // 2, w=z.shape[-1] // 2, ph=2, pw=2
        )


@torch.no_grad()
def native_prediction(base, z, t, text):
    from musubi_tuner.krea2.krea2_sampling import prepare

    x, pos, mask = prepare(z, text["text"].shape[1], 2, text["mask"].bool())
    y = base(img=x, context=text["text"], t=t, pos=pos, mask=mask)
    return rearrange(
        y, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=z.shape[-2] // 2, w=z.shape[-1] // 2, ph=2, pw=2
    )
