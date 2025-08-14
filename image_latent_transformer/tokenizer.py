import warnings
from functools import cached_property

from transformers import ByT5Tokenizer


class ByteTokenizer(ByT5Tokenizer):
    def __init__(self, *args, **kwargs):
        kwargs["bos_token"] = kwargs.get("bos_token", "<unk>")
        super().__init__(*args, **kwargs)

    def _add_eos_if_not_present(self, token_ids: list[int]) -> list[int]:
        token_ids = super()._add_eos_if_not_present(token_ids)

        # ByT5Tokenizer does not add BOS token by default, so we add it here

        if len(token_ids) > 0 and token_ids[0] == self.bos_token_id:
            warnings.warn(
                f"This sequence already has {self.bos_token_id}. In future versions this behavior may lead to "
                f"duplicated bos tokens being added.",
                stacklevel=2
            )
            return token_ids

        return  [self.bos_token_id] + token_ids

    @cached_property
    def stop_tokens(self) -> list[int]:
        """
        Get all stop tokens (EOS and whitespace) for this tokenizer.

        Returns:
            List of token IDs that should stop generation (EOS + all whitespace tokens)
        """
        # Get EOS token
        eos_token_id = self.eos_token_id

        # Get all whitespace token IDs
        whitespace_chars = [" ", "\t", "\r", "\n", "\v", "\f"]
        whitespace_token_ids = []

        for char in whitespace_chars:
            # Encode each whitespace character without special tokens
            tokens = self.encode(char, add_special_tokens=False)
            if tokens:  # Some whitespace might not have a token
                whitespace_token_ids.append(tokens[0])

        # Remove duplicates
        whitespace_token_ids = list(set(whitespace_token_ids))

        # Combine EOS and whitespace tokens
        return [eos_token_id] + whitespace_token_ids

if __name__ == "__main__":
    tokenizer = ByteTokenizer()

    tokenized = tokenizer(
        "a",
        return_tensors="pt",
        add_special_tokens=True  # Add BOS/EOS tokens
    )

    assert tokenized["input_ids"][0][0] == tokenizer.bos_token_id, "BOS token should be added at the start"
    assert tokenized["input_ids"][0][-1] == tokenizer.eos_token_id, "EOS token should be added at the end"
