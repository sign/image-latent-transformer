# Image Latent Transformer (ILT)

We believe that language modeling can be enhanced by treating text as images,
leveraging the visual structure of written language.

We perform whitespace segmentation and then encode each "token" as an image, and as a sequence of bytes.
The "tokens" go into a large transformer model, which predicts a representation of the next token.
Conditioned on this representation and all previous ones, we decode the next token as a sequence of bytes.

Finally, we render the predicted bytes as an image, which is then fed to the model as input for the next prediction.

![Model Architecture](./assets/architecture.png)


> [!NOTE]  
> For a more exhaustive writeup, please refer to our [paper draft](https://www.overleaf.com/read/rwxjzsvhrvwd#b9c65f).

## Environment

```shell
conda create -n ilt python=3.12 -y
conda activate ilt
conda install -c conda-forge pycairo pygobject manimpango -y
pip install ".[dev]"
```

> [!TIP]
> Run tests using `pytest` to ensure everything is working correctly.

## Model Setup

- **Bytes Encoder** - You can use any language model as the bytes encoder,
  such as causal LMs (e.g. `HuggingFaceTB/SmolLM2-135M`) or Masked LM (e.g. `answerdotai/ModernBERT-base`).
- **Image Encoder** - You can use any image encoder, from
  tiny (27m parameters; [Microsoft's SwinV2](https://huggingface.co/microsoft/swinv2-tiny-patch4-window16-256)) to
  large (272m parameters; [Apple's FastViT-HD](https://huggingface.co/kevin510/fast-vit-hd)).
- **Latent Transformer** - You can use any causal LM, such as `HuggingFaceTB/SmolLM2-360M`.
- **Bytes Decoder** - You can use any causal LM as the bytes decoder, such as `HuggingFaceTB/SmolLM2-135M`.

For language models, the parameter count is lower than reported, due to removing the embedding layers.

Our implementation allows for any mix-and-match. Some example setups are:

| Name   | Bytes Encoder                                                                | Image Encoder                                                                                             | Latent Transformer                                                    | Bytes Decoder                                                             | Total Parameters |
|--------|------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------|------------------|
| tiny   | [bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) (0.5m)               | [vit-tiny-patch16-224](https://huggingface.co/WinKawaks/vit-tiny-patch16-224) (5m)                        | [pythia-70m](https://huggingface.co/EleutherAI/pythia-70m) (19m)      | [tiny-lm](sbintuitions/tiny-lm) (3m)                                      | 28m              |
| small  | [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (111m) | [swinv2-tiny-patch4-window16-256](https://huggingface.co/microsoft/swinv2-tiny-patch4-window16-256) (27m) | [gemma-3-270m](https://huggingface.co/google/gemma-3-270m) (100m)     | [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (106m)  | 346m             |
| medium | [deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) (303m) | [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) (87m)                        | [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) (973m) | [gpt2-medium](https://huggingface.co/openai-community/gpt2-medium) (304m) | 1,674m           |

To turn off bytes encoding, set `bytes_encoder=False`, and similarly for images, set `image_encoder=False`.
You can also turn off a specific encoder after training has completed, for testing purposes.

> [!WARNING]  
> In implementation of the bytes decoder, we concatenate the embeddings of the bytes of the current token with
> all the embeddings of the previous tokens (on the word level). This is done since not all causal LMs support
> cross-attention, and so we want to avoid using it, and rely on the self-attention mechanism instead.

## Training

> [!NOTE]  
> Our generic collator `collate_fn` in [utils.py](./image_latent_transformer/utils.py) can be improved.

- [ ] Make our setup compatible with
  the [run_clm](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) example
  script, and train a small language model on
  the [Hebrew FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/heb_Hebr).


## Inference

> [!CAUTION]
> Our text [renderer.py](./image_latent_transformer/renderer.py) relying on the computer's font rendering capabilities.
> Rendering on different systems may yield different results.
> We call the community to create a more robust renderer, decoupled from the system's font rendering,
> for better consistency across platforms and easier reproducibility.

Since we have two decoders, the autoregressive prediction logic is a bit more complex than the usual,
and supporting decoding algorithms like beam-search is not trivial.

Thus, on the latent-transformer level, we only support greedy decoding for now.
On the bytes decoder level, we support all classical decoding algorithms supported by HuggingFace Transformers.

## Cite

If you use this code in your research, please consider citing the work:

```bibtex
@misc{moryossef2025ilt,
  title={Language Modeling with Text as Images},
  author={Moryossef, Amit},
  howpublished={\url{https://github.com/sign/image-latent-transformer}},
  year={2025}
}
```