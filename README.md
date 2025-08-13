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

To turn off bytes encoding, set `bytes_encoder=False`, and similarly for images, set `image_encoder=False`.
You can also turn off a specific encoder after training has completed, for testing purposes.


## Training 
> [!WARNING]  
> Our generic collator `collate_fn` in [utils.py](./image_latent_transformer/utils.py) was written using AI,
> and it may not work correctly in all scenarios.

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

Inference is not yet implemented. The autoregressive prediction logic is a bit more complex than the usual.
It is unnecessary to have inference if we just want to check the loss. 

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