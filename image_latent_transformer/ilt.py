import logging
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForMaskedLM
from transformers.modeling_outputs import CausalLMOutput

logger = logging.getLogger(__name__)


class ImageLatentTransformer(nn.Module):
    def __init__(self,
                 image_encoder: AutoModelForImageClassification,
                 bytes_encoder: Union[AutoModelForMaskedLM, AutoModelForCausalLM],
                 latent_transformer: AutoModelForCausalLM,
                 bytes_decoder: AutoModelForCausalLM,
                 padding_index: int = 0):
        super().__init__()

        assert bytes_encoder is not None or image_encoder is not None, "At least one encoder must be provided"

        self.image_encoder = image_encoder
        self.bytes_encoder = bytes_encoder
        self.latent_transformer = latent_transformer
        self.bytes_decoder = bytes_decoder
        self.padding_index = padding_index

        self.bytes_encoder_dim = bytes_encoder.config.hidden_size if bytes_encoder is not None else 0
        self.image_encoder_dim = image_encoder.config.hidden_size if image_encoder is not None else 0

        model_dim = latent_transformer.config.hidden_size
        self.encoder_mapping = nn.Linear(self.bytes_encoder_dim + self.image_encoder_dim, model_dim)
        self.decoder_mapping = nn.Linear(model_dim, bytes_decoder.config.hidden_size)

    def encode_images(self, input_pixels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_pixels: (BATCH, LENGTH, CHANNELS, HEIGHT, WIDTH)
        Returns:
            torch.Tensor: (BATCH, LENGTH, HIDDEN_DIM) - Image embeddings
        """
        B, L, C, H, W = input_pixels.shape  # noqa: N806

        # If image encoder is None, return zeros
        if self.image_encoder is None:
            return torch.zeros(B, L, self.image_encoder_dim, device=input_pixels.device, dtype=input_pixels.dtype)

        # Flatten batch and length dimensions
        input_pixels = input_pixels.view(B * L, C, H, W)
        # Encode images using the image encoder
        encoded_image = self.image_encoder(input_pixels, output_hidden_states=True)
        image_embeds = encoded_image.hidden_states[-1]  # Use the last hidden state
        # Pool channels dimension (e.g., mean pooling)
        image_embeds = image_embeds.mean(dim=1)
        return image_embeds.view(B, L, -1)

    def encode_texts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (BATCH, LENGTH, INPUT_TOKENS)
            attention_mask: (BATCH, LENGTH, INPUT_TOKENS)
        Returns:
            torch.Tensor: (BATCH, LENGTH, HIDDEN_DIM) - Text embeddings
        """
        B, L, T = input_ids.shape  # noqa: N806

        # If bytes encoder is None, return zeros
        if self.bytes_encoder is None:
            return torch.zeros(B, L, self.bytes_encoder_dim, device=input_ids.device, dtype=torch.float32)

        # Flatten batch and length dimensions
        input_ids = input_ids.view(B * L, T)
        attention_mask = attention_mask.view(B * L, T)
        # Encode texts using the bytes decoder as encoder
        text_outputs = self.bytes_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        text_embeds = text_outputs.hidden_states[-1]
        # Pool sequence dimension (e.g., mean pooling)
        text_embeds = text_embeds.mean(dim=1)  # (B*L, hidden_dim)
        return text_embeds.view(B, L, -1)

    def encode_input(self,
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     input_pixels: torch.Tensor):

        embeds = []
        if self.image_encoder_dim > 0:
            image_embeds = self.encode_images(input_pixels)
            logger.debug("Image embeddings shape: %s", image_embeds.shape)
            embeds.append(image_embeds)

        if self.bytes_encoder_dim > 0:
            text_embeds = self.encode_texts(input_ids, attention_mask)
            logger.debug("Text embeddings shape: %s", text_embeds.shape)
            embeds.append(text_embeds)

        assert len(embeds) > 0, "At least one type of encoder must be provided"

        concatenated_embeds = torch.cat(embeds, dim=-1)
        logger.debug("Concatenated embeddings shape: %s", concatenated_embeds.shape)

        return self.encoder_mapping(concatenated_embeds)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                input_pixels: torch.Tensor,
                labels_input: Optional[torch.Tensor] = None,
                labels_attention_mask: Optional[torch.Tensor] = None,
                labels_output: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids: (BATCH, LENGTH, INPUT_TOKENS)
            attention_mask: (BATCH, LENGTH, INPUT_TOKENS)
            input_pixels: (BATCH, LENGTH, CHANNELS, HEIGHT, WIDTH)
            labels_input: (BATCH, LENGTH, OUTPUT_TOKENS) - Input tokens for bytes decoder
            labels_attention_mask: (BATCH, LENGTH, OUTPUT_TOKENS) - Attention mask for labels
            labels_output: (BATCH, LENGTH, OUTPUT_TOKENS) - Target tokens for language modeling
        """

        # Embed images and texts
        mapped_embeds = self.encode_input(input_ids, attention_mask, input_pixels)

        # Process the sequence with the latent transformer
        latent_outputs = self.latent_transformer(inputs_embeds=mapped_embeds, output_hidden_states=True)
        latent_vectors = latent_outputs.hidden_states[-1]  # (B, L, hidden_dim)
        logger.debug("Latent vectors shape: %s", latent_vectors.shape)
        mapped_embeds = self.decoder_mapping(latent_vectors)
        logger.debug("Mapped embeddings shape: %s", mapped_embeds.shape)

        # Decode the latent vectors to bytes using parallel causal decoding
        logits = self.parallel_causal_decode(mapped_embeds, labels_input, labels_attention_mask)

        loss = None
        if labels_output is not None:
            # Flatten dimensions for cross entropy loss
            # logits: (B, L, T, vocab_size) -> (B*L*T, vocab_size)
            # labels_output: (B, L, T) -> (B*L*T,)
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = labels_output.reshape(-1)

            # Compute cross entropy loss
            loss = torch.nn.functional.cross_entropy(flat_logits, flat_labels, ignore_index=self.padding_index)

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=(mapped_embeds,),
            attentions=None
        )

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
