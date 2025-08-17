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
    --do_eval \
    --output_dir /tmp/test-clm \
    --logging_steps 1 \
    --logging_strategy steps \
    --max_steps 100 \
    --max_sequence_length 32 \
    --max_word_length 8
```

## Using Modal.com

Setup:

```bash
modal setup
```

Training:
```bash
modal run training/train_modal.py
```


