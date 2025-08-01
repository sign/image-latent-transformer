# Image Latent Transformer (ILT)

> ![NOTE]
> It would be nice if there was a good library to import, that handles the text rendering with all the necessary details, 
> with huge language support, and that is easy to use. That would make PIXELS experiments more accessible.

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
- [ ] Convert image_model.py to a tests file
- [ ] Create a dataset class to process images and text
- [ ] Create an overfitting test to see the model can be trained and sampled from
- [ ] Add a script to train the model on a small dataset
- [ ] Allow configuring the encoder to turn on/off modalities