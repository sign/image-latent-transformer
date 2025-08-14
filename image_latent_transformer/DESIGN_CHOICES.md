# Design Choices

### Bytes Decoder

Originally, we would have liked to use cross-attention in the bytes decoder,
however, not all causal language models support it.
Thus, we implemented a parallel causal decoding mechanism that prepends word-level latent vectors to the character
sequences, allowing us to use the self-attention mechanism instead.

At first, we added all previous latent vectors to the current character sequence, 
however, that proved difficult to properly implement in a parallel manner.
This is because we would also concatenate padding tokens, and the attention mask should change accordingly, 
as well as the position embeddings.

Our original implementation:

```pycon
def parallel_causal_decode(self,
                           latent_vectors: torch.Tensor,
                           target_ids: torch.Tensor,
                           target_mask: torch.Tensor) -> torch.Tensor:
    """
    Parallel causal decoding with word-level vectors prepended to character sequences.

    Args:
        latent_vectors: (B, L, hidden_dim) - latent representations for each word
        target_ids: (B, L, T) - target token IDs for each word
        target_mask: (B, L, T) - attention mask for target tokens

    Returns:
        torch.Tensor: (B, L, T, vocab_size) - logits for each token in each word
    """
    B, L, hidden_dim = latent_vectors.shape  # noqa: N806
    _, _, T = target_ids.shape  # noqa: N806

    # Step 1: Reshape target_ids from [B, L, T] to [B*L, T]
    target_ids_flat = target_ids.view(B * L, T)  # [B*L, T]
    target_mask_flat = target_mask.view(B * L, T)  # [B*L, T]

    # Step 2: Get embeddings for target tokens
    embed_layer = self.bytes_decoder.get_input_embeddings()
    target_embeds = embed_layer(target_ids_flat)  # [B*L, T, embed_dim]

    # Step 3: Create causal mask using tril to determine which words to prepend
    # For L=3 words, we want:
    # - 1st sequence: sees only word 0
    # - 2nd sequence: sees word 0 and word 1
    # - 3rd sequence: sees word 0, word 1, and word 2
    causal_mask = torch.tril(torch.ones(L, L, device=latent_vectors.device))  # [L, L]

    # Step 4: Reshape latent vectors for broadcasting
    latent_vectors_expanded = latent_vectors.unsqueeze(1).expand(B, L, L, hidden_dim)  # [B, L, L, hidden_dim]

    # Step 5: Apply causal mask to select which words each sequence sees
    # causal_mask[i, j] = 1 if sequence i can see word j
    causal_mask_expanded = causal_mask.unsqueeze(0).unsqueeze(-1).expand(B, L, L,
                                                                         hidden_dim)  # [B, L, L, hidden_dim]
    masked_latents = latent_vectors_expanded * causal_mask_expanded  # [B, L, L, hidden_dim]

    # Step 6: Reshape masked_latents to [B*L, L, hidden_dim]
    masked_latents_flat = masked_latents.view(B * L, L, hidden_dim)  # [B*L, L, hidden_dim]

    # Step 7: Concatenate word embeddings with character embeddings
    # Each sequence gets its corresponding words prepended
    combined_embeds = torch.cat([masked_latents_flat, target_embeds], dim=1)  # [B*L, L+T, embed_dim]

    # Step 8: Create attention mask
    # Word positions that are masked (0 in causal_mask) should have attention_mask=0
    causal_mask_flat = causal_mask.unsqueeze(0).expand(B, L, L).reshape(B * L, L)  # [B*L, L]
    combined_mask = torch.cat([causal_mask_flat, target_mask_flat], dim=1)  # [B*L, L+T]

    # Step 9: Pass through bytes decoder
    outputs = self.bytes_decoder(
        inputs_embeds=combined_embeds,
        attention_mask=combined_mask,
        output_hidden_states=False
    )

    # Step 10: Extract character-level logits (skip word positions)
    all_logits = outputs.logits  # [B*L, L+T, vocab_size]
    char_logits = all_logits[:, L:]  # [B*L, T, vocab_size]

    # Step 11: Reshape back to [B, L, T, vocab_size]
    logits = char_logits.view(B, L, T, -1)

    return logits
```