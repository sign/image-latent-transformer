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

## Training

> [!WARNING]  
> Our generic collator `collate_fn` in [utils.py](./image_latent_transformer/utils.py) was written using AI,
> and it may not work correctly in all scenarios.

Small scale Hebrew language modeling experiment.

TODO:

- [ ] Allow configuring the encoder to turn on/off modalities (only encode bytes, only encode images, etc.)
- [ ] Add a test for overfitting with image only or byte only inputs
- [ ] Add a test that the future does not leak into the past
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