from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from transformers import AutoModelForImageClassification, AutoModelForCausalLM, AutoModelForMaskedLM
from transformers.modeling_outputs import CausalLMOutput

logger = logging.getLogger(__name__)


class ImageLatentTransformer(nn.Module):
    def __init__(self,
                 image_encoder: AutoModelForImageClassification,
                 bytes_encoder: Union[AutoModelForMaskedLM, AutoModelForCausalLM],
                 latent_transformer: AutoModelForCausalLM,
                 bytes_decoder: AutoModelForCausalLM):
        super().__init__()
        self.image_encoder = image_encoder
        self.bytes_encoder = bytes_encoder
        self.latent_transformer = latent_transformer
        self.bytes_decoder = bytes_decoder

        embedding_dim = image_encoder.config.hidden_size + bytes_encoder.config.hidden_size
        model_dim = latent_transformer.config.hidden_size
        self.encoder_mapping = nn.Linear(embedding_dim, model_dim)
        self.decoder_mapping = nn.Linear(model_dim, bytes_decoder.config.hidden_size)

    def encode_images(self, input_pixels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_pixels: (BATCH, LENGTH, CHANNELS, HEIGHT, WIDTH)
        Returns:
            torch.Tensor: (BATCH, LENGTH, HIDDEN_DIM) - Image embeddings
        """

        # Flatten batch and length dimensions
        B, L, C, H, W = input_pixels.shape
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
        # Flatten batch and length dimensions
        B, L, T = input_ids.shape
        input_ids = input_ids.view(B * L, T)
        attention_mask = attention_mask.view(B * L, T)
        # Encode texts using the bytes decoder as encoder
        text_outputs = self.bytes_decoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        text_embeds = text_outputs.hidden_states[-1]
        # Pool sequence dimension (e.g., mean pooling)
        text_embeds = text_embeds.mean(dim=1)  # (B*L, hidden_dim)
        return text_embeds.view(B, L, -1)

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
        image_embeds = self.encode_images(input_pixels)
        logger.debug("Image embeddings shape: %s", image_embeds.shape)
        text_embeds = self.encode_texts(input_ids, attention_mask)
        logger.debug("Text embeddings shape: %s", text_embeds.shape)

        concatenated_embeds = torch.cat([image_embeds, text_embeds], dim=-1)
        logger.debug("Concatenated embeddings shape: %s", concatenated_embeds.shape)
        mapped_embeds = self.encoder_mapping(concatenated_embeds)

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
            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

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
        B, L, hidden_dim = latent_vectors.shape

        # Create cumulative latent contexts: each position sees all previous latent vectors
        # Use torch.tril to create lower triangular mask for causal dependency
        causal_mask = torch.tril(torch.ones(L, L, device=latent_vectors.device))

        # Create cumulative contexts efficiently using matrix multiplication
        # causal_mask[i, j] = 1 if j <= i, 0 otherwise
        # This gives us the cumulative sum for each position
        cumulative_latents = torch.matmul(causal_mask, latent_vectors)  # (B, L, hidden_dim)

        # Normalize by number of contributing vectors at each position
        position_counts = causal_mask.sum(dim=1, keepdim=True)  # (L, 1)
        cumulative_latents = cumulative_latents / position_counts.unsqueeze(0)

        # Prepare inputs for bytes decoder: shift target_ids right for teacher forcing
        shifted_target_ids = torch.cat([
            torch.zeros(B, L, 1, dtype=target_ids.dtype, device=target_ids.device),
            target_ids[:, :, :-1]
        ], dim=-1)

        shifted_mask = torch.cat([
            torch.ones(B, L, 1, dtype=target_mask.dtype, device=target_mask.device),
            target_mask[:, :, :-1]
        ], dim=-1)

        # Flatten for batch processing
        flat_latents = cumulative_latents.view(B * L, hidden_dim)
        flat_target_ids = shifted_target_ids.view(B * L, -1)
        flat_mask = shifted_mask.view(B * L, -1)

        # Use latent vectors as additional context by prepending them to embeddings
        target_embeds = self.bytes_decoder.get_input_embeddings()(flat_target_ids)
        latent_context = flat_latents.unsqueeze(1)  # (B*L, 1, hidden_dim)

        # Concatenate latent context with target embeddings
        contextualized_embeds = torch.cat([latent_context, target_embeds], dim=1)

        # Create attention mask for context + targets
        context_mask = torch.ones(B * L, 1, device=flat_mask.device)
        extended_mask = torch.cat([context_mask, flat_mask], dim=1)

        # Decode in parallel
        outputs = self.bytes_decoder(
            inputs_embeds=contextualized_embeds,
            attention_mask=extended_mask,
            output_hidden_states=True
        )

        # Extract logits for target positions (skip the latent context position)
        logits = outputs.logits[:, 1:, :]  # Remove first position (latent context)

        # Reshape back to original dimensions
        output_logits = logits.view(B, L, -1, logits.size(-1))

        return output_logits
