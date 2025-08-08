# Image Latent Transformer (ILT)

> [!NOTE]  
> It would be nice if there was a good library to import, that handles the text rendering with all the necessary details, 
> with huge language support, and that is easy to use. That would make PIXELS experiments more accessible.
> For the purpose of this project, we use our [custom renderer](./image_latent_transformer/renderer.py)

## Environment
```shell
conda create -n ilt python=3.12 -y
conda activate ilt
conda install -c conda-forge pycairo pygobject manimpango -y
pip install ".[dev]"
```

## Training

Small scale Hebrew language modeling experiment.

TODO: 
- [ ] Add a script to train the model on a small dataset
- [ ] Allow configuring the encoder to turn on/off modalities (only encode bytes, only encode images, etc.)
