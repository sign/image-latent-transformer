import collections.abc

# add imports near the top of your patch file
import types
from typing import Union

import torch
import transformers.models.vit.modeling_vit as hvit
from torch import nn
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTForImageClassification, ViTModel


def patch_vit_zero_pad_mask(model: ViTModel, pad_value: float = 0.0, tol: float = 0.0) -> None:
    """
    Make ViT invariant to zero-padding and zero out final hidden states at padded patches.

    Guarantees:
      1) Padded patches are excluded from attention via pre-softmax logits masking.
      2) Positional encodings for content patches are computed on the tight (content-only) grid,
         then scattered into the big sequence -> shared patches get identical pos-enc.
      3) At the end of the forward pass, hidden vectors at padded patches are set to 0 for:
           - outputs.last_hidden_state
           - outputs.hidden_states[-1] (if output_hidden_states=True)

    Assumptions:
      - Padding is zeros (tune `pad_value`/`tol` otherwise).
      - Content is a single top-left rectangle after padding (common case). If not, tell me and
        I’ll generalize the scatter to an arbitrary mask.
    """
    if getattr(model, "_zero_pad_mask_patched", False):
        return

    # -------- helpers --------
    def _patch_size_tuple(vit_model: ViTModel) -> tuple[int, int]:
        ps = vit_model.embeddings.patch_size
        if isinstance(ps, collections.abc.Iterable):
            ph, pw = int(ps[0]), int(ps[1])
        else:
            ph = pw = int(ps)
        return ph, pw

    def _valid_patch_grid(vit_model: ViTModel, pixel_values: torch.Tensor) -> tuple[torch.BoolTensor, int, int]:
        """
        Returns:
          valid_grid: (B, Hpatch, Wpatch) bool where True means "non-zero patch"
          Hpatch, Wpatch: patch grid size
        """
        B, C, H, W = pixel_values.shape
        ph, pw = _patch_size_tuple(vit_model)
        Hpatch, Wpatch = H // ph, W // pw
        patches = nn.functional.unfold(pixel_values, kernel_size=(ph, pw), stride=(ph, pw))  # (B, C*ph*pw, N)
        if tol > 0.0:
            zero_patch = (patches - pad_value).abs().le(tol).all(dim=1)  # (B, N)
        else:
            zero_patch = (patches == pad_value).all(dim=1)  # (B, N)
        valid_patch = ~zero_patch  # (B, N)
        return valid_patch.view(B, Hpatch, Wpatch), Hpatch, Wpatch

    def _compute_pairwise_token_mask(vit_model: ViTModel, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        (B,1,L,L) float mask: 1.0 allowed, 0.0 blocked. CLS always valid.
        """
        valid_grid, Hpatch, Wpatch = _valid_patch_grid(vit_model, pixel_values)
        B = valid_grid.size(0)
        valid_flat = valid_grid.view(B, -1)  # (B, N)
        cls = torch.ones(B, 1, dtype=torch.bool, device=valid_flat.device)
        valid_with_cls = torch.cat([cls, valid_flat], dim=1)  # (B, L)
        pair = (valid_with_cls[:, None, :, None] & valid_with_cls[:, None, None, :])  # (B,1,L,L)
        return pair.to(dtype=pixel_values.dtype)

    # -------- patch eager attention to support pre-softmax masking --------
    model.config._attn_implementation = "eager"
    if not hasattr(hvit, "_orig_eager_attention_forward"):
        hvit._orig_eager_attention_forward = hvit.eager_attention_forward

        def _eager_attention_forward_premask(
                module: nn.Module,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: torch.Tensor | None,
                scaling: float,
                dropout: float = 0.0,
                **kwargs,
        ):
            logits = torch.matmul(query, key.transpose(-1, -2)) * scaling  # (B,H,L,L)
            if getattr(module, "_use_pre_softmax_mask", False) and attention_mask is not None:
                mask_bool = attention_mask.bool() if attention_mask.dtype != torch.bool else attention_mask
                logits = logits.masked_fill(~mask_bool, torch.finfo(logits.dtype).min)
            attn = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
            if not getattr(module, "_use_pre_softmax_mask", False) and attention_mask is not None:
                attn = attn * attention_mask
            attn = nn.functional.dropout(attn, p=dropout, training=module.training)
            out = torch.matmul(attn, value).transpose(1, 2).contiguous()
            return out, attn

        hvit.eager_attention_forward = _eager_attention_forward_premask

    # -------- override ViTEmbeddings.forward to inject invariant positional encodings --------
    orig_embed_forward = model.embeddings.forward

    def embeddings_forward_with_invariant_pos(
            self: ViTEmbeddings,
            pixel_values: torch.Tensor,
            bool_masked_pos: torch.BoolTensor | None = None,
            interpolate_pos_encoding: bool | None = None,
    ) -> torch.Tensor:
        B, C, H, W = pixel_values.shape
        ph, pw = _patch_size_tuple(model)

        # Always allow arbitrary size; we'll handle positions ourselves
        patch_tokens = self.patch_embeddings(pixel_values, interpolate_pos_encoding=True)  # (B, N, D)
        N, D = patch_tokens.shape[1], patch_tokens.shape[2]

        # SimMIM mask token (remains valid)
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(B, N, -1)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            patch_tokens = patch_tokens * (1.0 - mask) + mask_tokens * mask

        # CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,D)

        # Tight content block per sample
        valid_grid, Hpatch, Wpatch = _valid_patch_grid(model, pixel_values)  # (B,Hpatch,Wpatch)
        row_has = valid_grid.any(dim=2)  # (B,Hpatch)
        col_has = valid_grid.any(dim=1)  # (B,Wpatch)

        def last_true_id(x: torch.Tensor) -> int:
            if not x.any():
                return 0
            idxs = torch.arange(x.numel(), device=x.device)
            return int((x.long() * idxs).max().item())

        # Per-sample interpolation (key change vs previous version)
        pos_big = torch.zeros(B, Hpatch, Wpatch, D, device=pixel_values.device, dtype=patch_tokens.dtype)
        cls_pos_all = torch.empty(B, 1, D, device=pixel_values.device, dtype=patch_tokens.dtype)
        for b in range(B):
            hs_b = last_true_id(row_has[b]) + 1
            ws_b = last_true_id(col_has[b]) + 1
            # Build dummy seq length for this sample
            dummy = torch.zeros(1, 1 + hs_b * ws_b, D, device=pixel_values.device, dtype=patch_tokens.dtype)
            pos_small = self.interpolate_pos_encoding(dummy, ph * hs_b, pw * ws_b)  # (1,1+hs_b*ws_b,D)
            cls_pos_all[b: b + 1] = pos_small[:, :1, :]  # (1,1,D)
            patch_pos_small = pos_small[:, 1:, :].reshape(1, hs_b, ws_b, D)  # (1,hs_b,ws_b,D)
            pos_big[b, :hs_b, :ws_b, :] = patch_pos_small[0]

        pos_big = pos_big.view(B, Hpatch * Wpatch, D)  # (B,N,D)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)  # (B,1+N,D)
        pos_cat = torch.cat([cls_pos_all, pos_big], dim=1)  # (B,1+N,D)
        tokens = tokens + pos_cat
        tokens = self.dropout(tokens)
        return tokens

    model.embeddings.forward = types.MethodType(embeddings_forward_with_invariant_pos, model.embeddings)

    # -------- wrap encoder.forward to merge masks + enable pre-softmax masking --------
    orig_enc_forward = model.encoder.forward

    def enc_forward_with_zero_pad_mask(self, hidden_states, head_mask=None, *args, **kwargs):
        token_mask = getattr(model, "_zero_pad_token_mask", None)  # (B,1,L,L) or None

        # Normalize head_mask => tensor or None
        if isinstance(head_mask, (list, tuple)):
            if len(head_mask) == 0 or all(h is None for h in head_mask):
                head_mask = None
            else:
                first = next(h for h in head_mask if h is not None)
                ones_like = torch.ones_like(first)
                head_mask = torch.stack([h if h is not None else ones_like for h in head_mask], dim=0)

        if token_mask is not None:
            num_layers = len(self.layer)
            B, _, L, _ = token_mask.shape
            H = model.config.num_attention_heads
            token_mask_full = token_mask.expand(B, H, L, L).unsqueeze(0).expand(num_layers, B, H, L, L)
            head_mask = token_mask_full if head_mask is None else (
                    head_mask.to(token_mask_full.dtype) * token_mask_full)

        # Enable pre-softmax masking
        for lyr in self.layer:
            lyr.attention.attention._use_pre_softmax_mask = True

        return orig_enc_forward(hidden_states, head_mask=head_mask, *args, **kwargs)

    model.encoder.forward = types.MethodType(enc_forward_with_zero_pad_mask, model.encoder)

    # -------- wrap model.forward: compute mask, call orig, then zero-out padded outputs --------
    orig_model_forward = model.forward

    def forward_with_zero_pad_mask(self: ViTModel, *args, **kwargs):
        pixel_values = kwargs.get("pixel_values", None)
        if pixel_values is None and len(args) > 0:
            pixel_values = args[0]

        # Build masks for this call
        valid_grid, Hpatch, Wpatch = _valid_patch_grid(self, pixel_values) if pixel_values is not None else (None, None,
                                                                                                             None)
        self._zero_pad_token_mask = _compute_pairwise_token_mask(self,
                                                                 pixel_values) if pixel_values is not None else None

        # Run original forward
        outputs = orig_model_forward(*args, **kwargs)

        # Post-mask: zero-out outputs
        if valid_grid is not None:
            B = valid_grid.size(0)
            valid_flat = valid_grid.view(B, -1)  # (B, N)

            # Fully zero images in batch
            fully_zero = ~valid_flat.any(dim=1)  # (B,)
            if fully_zero.any():
                # last_hidden_state
                if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    outputs.last_hidden_state[fully_zero] = 0
                # hidden_states: zero ALL layers for those samples to avoid NaN propagation
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None and len(
                        outputs.hidden_states) > 0:
                    hs = list(outputs.hidden_states)
                    for i in range(len(hs)):
                        hs[i][fully_zero] = 0
                    outputs.hidden_states = tuple(hs)
                # pooler_output (BaseModelOutputWithPooling)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    outputs.pooler_output[fully_zero] = 0
                # logits (if you patch a ViTForImageClassification)
                if hasattr(outputs, "logits") and outputs.logits is not None:
                    outputs.logits[fully_zero] = 0

            # Partially padded samples: zero only padded patches (CLS stays)
            non_empty = ~fully_zero
            if non_empty.any():
                keep = torch.cat(
                    [torch.ones(B, 1, dtype=torch.bool, device=valid_flat.device), valid_flat],
                    dim=1,
                )  # (B, L) with CLS=True
                keep = keep[:, :, None]  # (B, L, 1)

                # last_hidden_state
                if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    outputs.last_hidden_state[non_empty] = (
                            outputs.last_hidden_state[non_empty] * keep[non_empty].to(outputs.last_hidden_state.dtype)
                    )

                # hidden_states[-1] only (leave earlier for debugging/consistency)
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None and len(
                        outputs.hidden_states) > 0:
                    hs = list(outputs.hidden_states)
                    hs[-1][non_empty] = hs[-1][non_empty] * keep[non_empty].to(hs[-1].dtype)
                    outputs.hidden_states = tuple(hs)

        # cleanup
        if hasattr(self, "_zero_pad_token_mask"):
            delattr(self, "_zero_pad_token_mask")

        return outputs

    model.forward = types.MethodType(forward_with_zero_pad_mask, model)
    model._zero_pad_mask_patched = True


def maybe_patch_vit_model(model: Union[ViTModel, ViTForImageClassification]) -> None:
    vit = model.vit if hasattr(model, "vit") else model
    if not getattr(vit, "_zero_pad_token_mask", None):
        patch_vit_zero_pad_mask(vit)


def is_vit_model(model: nn.Module) -> bool:
    return isinstance(model, ViTModel) or isinstance(getattr(model, "vit", None), ViTModel)
