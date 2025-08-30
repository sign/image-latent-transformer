FROM mambaorg/micromamba:latest

# Apt install happens as root
USER root
RUN apt-get update && apt-get install -y git

# Install python version
RUN micromamba install python=3.12 && micromamba clean -a -y

# Install rendering dependencies
RUN micromamba install -y pycairo pygobject manimpango -c conda-forge && micromamba clean -a -y

# Install package dependencies
RUN mkdir -p /app/image_latent_transformer/tokenizer
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN micromamba run pip install ".[train]"

# Make Python discoverable for Modal
RUN ln -s /opt/conda/bin/python /usr/local/bin/python

USER $MAMBA_USER

# docker build -t ilt .
