import warnings
from typing import Union, Optional

from transformers import PretrainedConfig, AutoConfig


class ImageLatentTransformerConfig(PretrainedConfig):
    model_type = "image_latent_transformer"

    sub_configs = {
        "image_encoder": AutoConfig,
        "bytes_encoder": AutoConfig,
        "latent_transformer": AutoConfig,
        "bytes_decoder": AutoConfig,
    }

    def __init__(self,
                 image_encoder: Optional[Union[AutoConfig, dict]] = None,
                 bytes_encoder: Optional[Union[AutoConfig, dict]] = None,
                 latent_transformer: Union[AutoConfig, dict] = None,
                 bytes_decoder: Union[AutoConfig, dict] = None,
                 modality_dropout: float = 0.15,
                 num_tokens: int = 256,
                 **kwargs):
        super().__init__(**kwargs)

        assert bytes_encoder is not None or image_encoder is not None, "At least one encoder must be provided"

        self.image_encoder = image_encoder
        self.bytes_encoder = bytes_encoder
        self.latent_transformer = latent_transformer
        self.bytes_decoder = bytes_decoder
        self.modality_dropout = modality_dropout
        self.num_tokens = num_tokens

        if image_encoder is None or bytes_encoder is None:
            warnings.warn("Image encoder and bytes encoder are not provided, setting modality_dropout to 0.0")
            self.modality_dropout = 0.0

        super().__init__(is_decoder=True, **kwargs)
