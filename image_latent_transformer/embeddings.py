from typing import Union

import torch
import torch.nn as nn
from torch.nn import Embedding
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM


def unpack_bits(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.uint8:
        x = x.to(torch.uint8)
    shifts = torch.arange(7, -1, -1, device=x.device, dtype=torch.uint8)
    return ((x.unsqueeze(-1) >> shifts) & 1).to(torch.float32)  # (B, L, 8)


class PatchedBitEmbeddings(nn.Module):
    """
    Wraps an existing nn.Embedding and adds an 8->D bit projection.
    Exposes .weight to preserve HF weight-tying semantics.
    """

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding
        self.bit_proj = nn.Linear(8, embedding.embedding_dim, bias=False)
        nn.init.zeros_(self.bit_proj.weight)  # start with no bit contribution
        
        # Create the combined weight parameter, once
        all_bytes = torch.arange(256, dtype=torch.long, device=embedding.weight.device)
        base_embeds = embedding(all_bytes)  # (256, D)
        bits = unpack_bits(all_bytes)  # (256, 8)
        bit_feats = self.bit_proj(bits)  # (256, D)
        combined_weight = base_embeds + bit_feats
        
        self.weight = nn.Parameter(combined_weight, requires_grad=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Use the combined weight like a normal embedding
        return nn.functional.embedding(input_ids, self.weight)


def patch_embedding_layer(model: Union[AutoModelForMaskedLM, AutoModelForCausalLM]):
    embeddings: Embedding = model.get_input_embeddings()
    assert isinstance(embeddings, Embedding), "Expected nn.Embedding layer"
    assert len(embeddings.weight) == 256, "Expected byte-level embedding layer"
    patched_embeddings = PatchedBitEmbeddings(embeddings)
    model.set_input_embeddings(patched_embeddings)

    model.tie_embeddings_and_encoder_decoder()


def join_embedding_layer(model: Union[AutoModelForMaskedLM, AutoModelForCausalLM]):
    embeddings: PatchedBitEmbeddings = model.get_input_embeddings()
    assert isinstance(embeddings, PatchedBitEmbeddings), "Expected patched embedding layer"

    # Reuse the original embedding to preserve weight tying
    original_embedding = embeddings.embedding
    original_embedding.weight.data = embeddings.weight.data
    model.set_input_embeddings(original_embedding)

    model.tie_embeddings_and_encoder_decoder()
