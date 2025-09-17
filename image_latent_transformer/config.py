from typing import Optional, Union

from transformers import CONFIG_MAPPING, AutoConfig, PretrainedConfig


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

        for name in self.sub_configs.keys():
            self.init_sub_config(name)

        self.modality_dropout = modality_dropout
        self.num_tokens = num_tokens

    def init_sub_config(self, name: str):
        config = getattr(self, name, None)
        if isinstance(config, dict):
            model_type = config.get("model_type", None)
            config_cls = CONFIG_MAPPING[model_type] if model_type else PretrainedConfig
            config = config_cls(**config)
            setattr(self, name, config)

        if config is None:
            setattr(self, name, PretrainedConfig())
