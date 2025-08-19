import warnings
from typing import Union, Optional

from transformers import PretrainedConfig, AutoConfig, CONFIG_MAPPING


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
        # Configuration defaults
        kwargs["is_decoder"] = kwargs.get("is_decoder", True)
        super().__init__(**kwargs)

        self.image_encoder = image_encoder
        self.bytes_encoder = bytes_encoder
        self.latent_transformer = latent_transformer
        self.bytes_decoder = bytes_decoder

        torch_dtype = kwargs.get("torch_dtype", None)
        self.fix_sub_config("image_encoder", torch_dtype=torch_dtype)
        self.fix_sub_config("bytes_encoder", torch_dtype=torch_dtype)
        self.fix_sub_config("latent_transformer", torch_dtype=torch_dtype)
        self.fix_sub_config("bytes_decoder", torch_dtype=torch_dtype)

        self.modality_dropout = modality_dropout
        self.num_tokens = num_tokens

        if image_encoder is None or bytes_encoder is None:
            warnings.warn("Image encoder and bytes encoder are not provided, setting modality_dropout to 0.0")
            self.modality_dropout = 0.0


    def fix_sub_config(self, name: str, torch_dtype=None):
        config = getattr(self, name, None)
        if isinstance(config, dict):
            model_type = getattr(config, "model_type", None)
            config_cls = CONFIG_MAPPING[model_type] if model_type else PretrainedConfig
            config = config_cls(**config)
            setattr(self, name, config)
        if torch_dtype is not None:
            config.torch_dtype = torch_dtype
        setattr(self, name, config)
