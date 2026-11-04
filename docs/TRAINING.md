# Training guide

1. Run `python scripts/detect_hardware.py`.
2. Prepare UTF-8 text files in `datasets/corpus/`.
3. Record sources and licenses in a dataset manifest.
4. Train the tokenizer on those same sources.
5. Set `vocab_size` in a model config to the tokenizer's actual vocabulary size.
6. Ensure the corpus is long enough for at least one train and one validation block.

Train the tokenizer:

```bash
python scripts/train_tokenizer.py --input-dir datasets/corpus --output-dir artifacts/tokenizer --vocab-size 2048
```

Train the micro-model:

```bash
python scripts/train.py --data-dir datasets/corpus --tokenizer artifacts/tokenizer/tokenizer.model --config configs/micro.json --output-dir artifacts/micro-run --max-steps 100 --batch-size 2
```

The baseline prioritizes correctness over speed. A completed run only means the requested optimization steps executed. It does not establish useful language ability. Review the validation loss trend, Arabic and English generations, training token count, and actual hardware. A randomly initialized model initially generates poor text.

Do not start long jobs until memory requirements are understood. Back up checkpoints and never commit private corpora or large weights to Git.
