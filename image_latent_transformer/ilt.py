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
        # Pool sequence dimension weighted by attention mask
        sequence_lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
        text_embeds = text_embeds.sum(dim=1) / sequence_lengths
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

    def _num_words_per_datum(self, attention_mask: torch.Tensor) -> torch.Tensor:
        return attention_mask.sum(dim=-1).gt(0).sum(dim=-1)

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

        # Step 3: Each decoder uses only one latent vector (no history)
        # Decoder i uses latent_vectors[:, i]
        # Reshape from [B, L, hidden_dim] to [B*L, hidden_dim] then add sequence dimension
        latent_vectors_flat = latent_vectors.view(B * L, hidden_dim).unsqueeze(1)  # [B*L, 1, hidden_dim]

        # Step 4: Concatenate single latent vector with character embeddings
        # Each sequence gets only its corresponding latent vector prepended
        combined_embeds = torch.cat([latent_vectors_flat, target_embeds], dim=1)  # [B*L, 1+T, embed_dim]

        # Step 5: Create attention mask
        # Each decoder only sees its single latent vector, so mask is all ones for the single latent position
        latent_mask = torch.ones(B * L, 1, device=latent_vectors.device)  # [B*L, 1]
        combined_mask = torch.cat([latent_mask, target_mask_flat], dim=1)  # [B*L, 1+T]

        # Step 6: Pass through bytes decoder
        outputs = self.bytes_decoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=False
        )

        # Step 7: Extract character-level logits (skip the single latent position)
        all_logits = outputs.logits  # [B*L, 1+T, vocab_size]
        char_logits = all_logits[:, 1:]  # [B*L, T, vocab_size]

        # Step 8: Reshape back to [B, L, T, vocab_size]
        logits = char_logits.view(B, L, T, -1)

        return logits
