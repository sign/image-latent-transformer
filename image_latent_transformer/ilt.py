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
            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=0) # 0 is the padding index

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
        Parallel causal decoding with word-level vectors followed by character-level vectors.
        Creates a sequence like: word1, word2, ..., wordL, char1, char2, ..., charT
        
        Args:
            latent_vectors: (B, L, hidden_dim) - latent representations for each word
            target_ids: (B, L, T) - target token IDs for each word
            target_mask: (B, L, T) - attention mask for target tokens
            
        Returns:
            torch.Tensor: (B, L, T, vocab_size) - logits for each token in each word
        """
        B, L, hidden_dim = latent_vectors.shape
        _, _, T = target_ids.shape
        
        # Get embeddings for target tokens
        embed_layer = self.bytes_decoder.get_input_embeddings()
        
        # Get target token embeddings
        target_embeds = embed_layer(target_ids)  # (B, L, T, embed_dim)
        
        # Flatten target embeddings from (B, L, T, embed_dim) to (B, L*T, embed_dim)
        target_embeds_flat = target_embeds.view(B, L * T, -1)
        
        # Concatenate latent vectors and target embeddings
        # First L positions are latent vectors, next L*T positions are target tokens
        combined_embeds = torch.cat([latent_vectors, target_embeds_flat], dim=1)  # (B, L + L*T, embed_dim)
        
        # Flatten target_mask from (B, L, T) to (B, L*T)
        target_mask_flat = target_mask.view(B, L * T)
        
        # Create combined attention mask
        # Word positions are always attended (1s), character positions use target_mask
        word_mask = torch.ones(B, L, dtype=torch.bool, device=latent_vectors.device)
        combined_mask = torch.cat([word_mask, target_mask_flat], dim=1)  # (B, L + L*T)
        
        logger.debug("Combined embeds shape: %s", combined_embeds.shape)
        logger.debug("Combined mask shape: %s", combined_mask.shape)
        logger.debug("Target IDs shape: %s", target_ids.shape)
        logger.debug("Target mask shape: %s", target_mask.shape)

        # Pass through bytes decoder
        outputs = self.bytes_decoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=False
        )

        # Extract only the character-level logits (skip word-level outputs)
        all_logits = outputs.logits  # (B, L + L*T, vocab_size)
        char_logits = all_logits[:, L:]  # (B, L*T, vocab_size)

        # Reshape back to (B, L, T, vocab_size)
        logits = char_logits.view(B, L, T, -1)

        return logits