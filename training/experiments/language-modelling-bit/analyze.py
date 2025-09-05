from huggingface_hub import load_state_dict_from_file
from transformers import AutoModelForCausalLM, AutoConfig

from image_latent_transformer.embeddings import patch_embedding_layers

MODEL_CHECKPOINT = "./output-clm-byte-bit/checkpoint-5000"

# Load model
config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForCausalLM.from_config(config)
patch_embedding_layers(model)

state_dict = load_state_dict_from_file(f"{MODEL_CHECKPOINT}/model.safetensors")
model.load_state_dict(state_dict)

# Inspect bit projection weights
embeddings = model.get_input_embeddings()
print(embeddings.bit_proj_w.data)
