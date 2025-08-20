# Training


Setup with:

```bash
pip install ".[train]"
```

Run:
[//]: # (TODO: Unclear why `remove_unused_columns=False` is needed, but it is required to avoid errors during training.)
```bash
python -m training.train \
    --image_encoder_model_name_or_path "WinKawaks/vit-tiny-patch16-224" \
    --bytes_encoder_model_name_or_path "prajjwal1/bert-tiny" \
    --latent_transformer_model_name_or_path "EleutherAI/pythia-70m" \
    --bytes_decoder_model_name_or_path "sbintuitions/tiny-lm" \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --remove_unused_columns False \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --save_steps 10 \
    --output_dir training/output \
    --logging_steps 1 \
    --logging_strategy steps \
    --max_steps 50 \
    --max_sequence_length 32 \
    --max_word_length 8 \
    --dataloader_num_workers 4 \
    --include_tokens_per_second True \
    --include_num_input_tokens_seen True \
    --max_train_samples 16
```

## Using Modal.com

Setup:

```bash
modal setup
```

Training:
```bash
modal run -m training.modal::train
modal run -m training.modal::predict   
```

Download trained model:
```bash
mkdir -p output
modal volume get model-output / training/output
```

### Training Quirks

`num_input_tokens_seen` and `train_tokens_per_second` are calculated based on the number of bytes the model decodes.
That means that in practice, if `max_word_length=32`, a rough estimate of
the real number of **words** the model sees should be divided by 32.
