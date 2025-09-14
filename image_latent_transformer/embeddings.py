import torch
import torch.nn as nn
from torch.nn import Embedding


def unpack_bits(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8, "Expected bytes tensor input (torch.uint8)"
    shifts = torch.arange(7, -1, -1, device=x.device, dtype=torch.uint8)
    shifted = x.unsqueeze(-1) >> shifts  # (B, L, 8)
    return shifted & 1


class PatchedBitEmbeddings(nn.Module):
    """
    nn.Embedding + additive 8->D bit projection.
    Exposes a *Parameter* `.weight` for HF tying, but learns via:
      - base table: self.embeddings.weight (256, D)
      - bit proj  : self.bit_proj_w       (D, 8)

    The façade `self.weight` is refreshed in-place with:
        W = base + bits(256,8) @ bit_proj_w.T(8,D)
    and all grads arriving on the façade are re-routed to the real params.
    """

    def __init__(self, embeddings: nn.Embedding):
        super().__init__()
        assert isinstance(embeddings, nn.Embedding)
        assert embeddings.weight.shape[0] == 256, "Expected byte-level embedding layer (256 rows)."

        self.embeddings = embeddings
        D = embeddings.embedding_dim  # noqa: N806
        dtype = self.embeddings.weight.dtype

        # Tiny bit projection; use a bare Parameter to avoid Module overhead & extra .t()
        self.bit_proj_w = nn.Parameter(torch.zeros(D, 8, dtype=dtype))  # init=0 ⇒ starts identical to base table

        # Tieable façade parameter (what HF ties to the LM head)
        self.weight = nn.Parameter(embeddings.weight.detach().clone(), requires_grad=True)

        # Bits table buffer (float32 initially); device/dtype-adjusted copies are cached lazily
        all_bytes = torch.arange(256, dtype=torch.uint8)
        self.register_buffer("_bits256_base", unpack_bits(all_bytes), persistent=False)
        self._bits_cached = None  # (256, 8) on current device/dtype
        self._bits_device = torch.device("meta")  # sentinel (forces first refresh)
        self._bits_dtype = dtype

        # Rebuild only when needed (params changed or device/dtype changed)
        self._last_base_v = -1
        self._last_bit_v = -1
        self._last_device = None
        self._last_dtype = None

        # Route grads from façade into the true params; block façade updates
        def _route_grad(grad: torch.Tensor):
            # grad = dL/dW, shape (256, D)
            g = grad.detach()

            # base grad: += grad
            ew = self.embeddings.weight
            if ew.grad is None:
                ew.grad = g.clone()
            else:
                ew.grad.add_(g)

            # bit proj grad: dL/d(D,8) = (dL/d(256,D))^T @ bits(256,8) → (D,8)
            # use cached device/dtype-adjusted bits
            self._ensure_bits_cached()  # cheap no-op after first call
            gb = g.t().mm(self._bits_cached)  # (D,8)
            if self.bit_proj_w.grad is None:
                self.bit_proj_w.grad = gb.clone()
            else:
                self.bit_proj_w.grad.add_(gb)

            # façade param should not be optimized directly
            return torch.zeros_like(grad)

        self.weight.register_hook(_route_grad)

        # Keep façade fresh even if call order is quirky (runs before forward)
        self.register_forward_pre_hook(lambda m, inp: m._maybe_refresh_weight_())

    # ---- internals ----

    def _ensure_bits_cached(self):
        """Cache bits in current device/dtype (no work in the hot path after first time)."""
        w = self.embeddings.weight
        if (self._bits_device is not w.device) or (self._bits_dtype != w.dtype):
            self._bits_cached = self._bits256_base.to(device=w.device, dtype=w.dtype, non_blocking=True).contiguous()
            self._bits_device, self._bits_dtype = w.device, w.dtype

    def _needs_refresh(self) -> bool:
        w = self.embeddings.weight
        bw = self.bit_proj_w
        return (
                w._version != self._last_base_v
                or bw._version != self._last_bit_v
                or w.device is not self._last_device
                or w.dtype != self._last_dtype
        )

    def _mark_refreshed(self):
        w = self.embeddings.weight
        self._last_base_v = w._version
        self._last_bit_v = self.bit_proj_w._version
        self._last_device = w.device
        self._last_dtype = w.dtype

    @torch.no_grad()
    def _refresh_weight_(self):
        """façade = base + bits @ bit_proj^T  (fused with addmm_ for speed)."""
        self._ensure_bits_cached()
        # Copy base
        self.weight.data.copy_(self.embeddings.weight)
        # Fused GEMM + add: (256,8) @ (8,D) → (256,D)
        self.weight.data.addmm_(self._bits_cached, self.bit_proj_w.t())

    @torch.no_grad()
    def _maybe_refresh_weight_(self):
        if self._needs_refresh():
            self._refresh_weight_()
            self._mark_refreshed()

    # ---- public API ----

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Façade already refreshed by pre-hook; forward is just a normal embedding lookup
        input_ids = input_ids.to(dtype=torch.long)
        # TODO: ideally, we should use
        #  return nn.functional.embedding(input_ids, self.weight)
        #  https://github.com/pytorch/pytorch/issues/162918
        return self.weight[input_ids]


def patch_embedding_layers(model):
    embeddings: Embedding = model.get_input_embeddings()
    assert isinstance(embeddings, Embedding), "Expected nn.Embedding layer"
    assert len(embeddings.weight) == 256, "Expected byte-level embedding layer"
    patched_embeddings = PatchedBitEmbeddings(embeddings)

    model.set_input_embeddings(patched_embeddings)
    model.tie_embeddings_and_encoder_decoder()


def join_embedding_layers(model):
    embeddings: PatchedBitEmbeddings = model.get_input_embeddings()
    assert isinstance(embeddings, PatchedBitEmbeddings), "Expected patched embedding layer"

    # Reuse the original embedding to preserve weight tying
    original_embedding = embeddings.embeddings
    original_embedding.weight.data = embeddings.weight.data

    model.set_input_embeddings(original_embedding)
    model.tie_embeddings_and_encoder_decoder()
