import logging
from typing import Any, Optional

import torch
from transformers import AutoImageProcessor
from transformers.generation import GenerationConfig

from image_latent_transformer.ilt import ImageLatentTransformer
from image_latent_transformer.renderer import deconstruct_images, render_texts, render_texts_torch
from image_latent_transformer.tokenizer import ByteTokenizer
from image_latent_transformer.utils import collate_images

logger = logging.getLogger(__name__)


class ImageLatentTransformerForTextGeneration(ImageLatentTransformer):
    """
    ImageLatentTransformer with iterative text generation capabilities.

    This class extends the base ImageLatentTransformer to generate text
    word-by-word using a two-level approach: latent transformer provides
    context, bytes decoder generates individual words.
    """

    def _generate_latents(self, latent_past_key_values, encoded_input: torch.Tensor,
                          attention_mask: torch.Tensor,
                          num_words: torch.Tensor) -> tuple[Any, torch.Tensor]:
        latent_output = self.latent_transformer(
            inputs_embeds=encoded_input,
            attention_mask=attention_mask,
            # TODO: past_key_values can improve efficiency, but currently fails tests.
            # past_key_values=latent_past_key_values,
            use_cache=True,
            output_hidden_states=True
        )

        # Get the last latent state for the new word
        latents = latent_output.hidden_states[-1]  # (B, L, hidden_dim)

        assert len(latents.shape) == 3, "Latents should be of shape (B, L, latent_dim)"

        batch_indices = torch.arange(latents.size(0), device=latents.device)
        last_latents = latents[batch_indices, num_words - 1].unsqueeze(1)  # (B, 1, latent_dim)

        # Map latent state to bytes decoder dimension
        mapped_latent = self.decoder_mapping(last_latents)  # (B, 1, bytes_decoder_dim)

        return latent_output.past_key_values, mapped_latent

    def _generate_word_bytes(
            self,
            latents: torch.Tensor,
            tokenizer: ByteTokenizer,
            max_word_length: Optional[int] = None,
            bytes_generation_config: Optional[GenerationConfig] = None,
            **bytes_generation_kwargs
    ) -> torch.Tensor:
        # Prepare generation config
        if bytes_generation_config is None:
            bytes_generation_config = GenerationConfig(
                max_new_tokens=max_word_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.stop_tokens,
            )

        # Add stop tokens to generation config
        bytes_generation_config.eos_token_id = tokenizer.stop_tokens

        # Start generation with BOS token
        current_input_ids = torch.full((len(latents), 1), tokenizer.bos_token_id,
                                       dtype=torch.long, device=latents.device)
        current_input_embeds = self.bytes_decoder.get_input_embeddings()(current_input_ids)

        inputs_embeds = torch.cat([latents, current_input_embeds], dim=1)  # (B, 2, bytes_decoder_dim)

        # Generate bytes for this word
        return self.bytes_decoder.generate(
            inputs_embeds=inputs_embeds,
            generation_config=bytes_generation_config,
            **bytes_generation_kwargs
        )

    def prepare_next_inputs(self, words: list[str],
                            words_indexes: torch.Tensor,
                            tokenizer: ByteTokenizer,
                            image_processor: AutoImageProcessor,
                            encoded_input: torch.Tensor) -> torch.Tensor:
        tokenized_words = tokenizer(
            words,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        )
        new_input_ids = tokenized_words["input_ids"].to(encoded_input.device)
        new_attention_mask = tokenized_words["attention_mask"].to(encoded_input.device)

        # Render images for the new words, one per batch item
        patch_size = self.image_encoder.config.patch_size if self.image_encoder else 16
        images = [[image] for image in render_texts_torch(words, image_processor)]
        pixel_values, pixel_mask = collate_images(images, patch_size=patch_size)
        new_input_pixels = pixel_values.to(encoded_input.device)
        new_input_pixels_mask = pixel_mask.to(encoded_input.device)

        new_encoded_input = self.encode_input(new_input_ids.unsqueeze(1),
                                              new_attention_mask.unsqueeze(1),
                                              new_input_pixels,
                                              new_input_pixels_mask)

        # Extend the encoded input with an empty tensor
        empty_word = torch.zeros_like(encoded_input[:, 0:1])
        encoded_input = torch.cat([encoded_input, empty_word], dim=1)

        # Apply the new words' indexes to the encoded input
        batch_indices = torch.arange(encoded_input.size(0), device=encoded_input.device)
        encoded_input[batch_indices, words_indexes] = new_encoded_input.squeeze(1)

        return encoded_input

    @torch.no_grad()
    def generate(
            self,
            input_pixels: torch.Tensor,
            input_pixels_mask: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            tokenizer: ByteTokenizer,
            image_processor: AutoImageProcessor,
            max_generated_words: int = 50,
            max_word_length: int = None,
            bytes_generation_config: Optional[GenerationConfig] = None):
        """
        Generate text sequences using iterative latent-then-bytes generation.

        This method implements a two-level generation process:
        1. Latent transformer generates latent states (greedy only for now)
        2. Bytes decoder generates words using full HuggingFace generation capabilities

        In practice, this means:
        1. self.encode_input(input_ids, attention_mask, input_pixels)
        2. self.latent_transformer.generate(...) to get latent states
        3. self.bytes_decoder.generate(...) to generate a single word from latent states (sequence of bytes)
        4. Detokenize the generated bytes to get the final word
        5. Create a new input_ids tensor for the new bytes. new attention_mask,
            and an input_pixels tensor only for the new word (using render_texts())
        6. Call self.encode_input() again with the new input_ids, attention_mask, and input_pixels
        7. Concatenate the encoded input with the previous encoded input to form the
            new input for the next word generation
        8. Repeat from step 2 until max_generated_words is reached, or the generated word is just an EOS token.

        We utilize the HuggingFace generation cache to efficiently generate.

        Args:
            input_pixels: Image inputs (B, L, C, H, W)
            input_ids: Text input tokens (B, L, T) for encoding
            attention_mask: Attention mask for text inputs (B, L, T)
            tokenizer: ByteTokenizer instance
            image_processor: AutoImageProcessor for processing images
            max_generated_words: Maximum number of words to generate
            max_word_length: Maximum number of characters to generate per word
            bytes_generation_config: Generation config for bytes_decoder
        """
        num_words = self._num_words_per_datum(attention_mask)

        # Track generated words per sample
        all_generated_words = [[] for _ in range(len(input_pixels))]

        # Step 1: Encode initial input
        encoded_input = self.encode_input(input_ids, attention_mask,
                                          input_pixels, input_pixels_mask)

        past_key_values = None  # Initialize past key values for caching

        # Main generation loop
        for _ in range(max_generated_words):
            # Step 2: Generate next latent state
            words_attention_mask = self._words_sequence_attention_mask(num_words)
            past_key_values, latents = self._generate_latents(past_key_values, encoded_input,
                                                              attention_mask=words_attention_mask,
                                                              num_words=num_words)

            # Step 3: Generate bytes for this word
            generated_bytes = self._generate_word_bytes(
                latents=latents,
                tokenizer=tokenizer,
                max_word_length=max_word_length,
                bytes_generation_config=bytes_generation_config
            )

            # Step 4: Process generated bytes
            words = tokenizer.batch_decode(generated_bytes, skip_special_tokens=True)
            batch_finished = all(len(word) == 0 for word in words)

            if batch_finished:
                break

            for word, generated_words in zip(words, all_generated_words):
                if len(generated_words) == 0 or len(generated_words[-1]) > 0:
                    # Only add words if not EOS
                    generated_words.append(word)

            # Step 5, 6, 7: Create next inputs and encode them\
            encoded_input = self.prepare_next_inputs(words, num_words, tokenizer, image_processor, encoded_input)

            num_words += 1

        # Step 8: Return generated texts
        texts = ["".join(generated_words) for generated_words in all_generated_words]
        return texts
