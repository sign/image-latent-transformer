# Image Latent Transformer (ILT)

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

- [ ] Make our setup compatible with
  the [run_clm](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) example
  script, and train a small language model on
  the [Hebrew FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/heb_Hebr).
- [ ] Allow configuring the encoder to turn on/off modalities (only encode bytes, only encode images, etc.)

## Inference

> [!CAUTION]
> Our text [renderer.py](./image_latent_transformer/renderer.py) relying on the computer's font rendering capabilities.
> Rendering on different systems may yield different results.
> We call the community to create a more robust renderer, decoupled from the system's font rendering,
> for better consistency across platforms and easier reproducibility.

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