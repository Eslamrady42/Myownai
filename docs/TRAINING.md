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

## First bilingual corpus run on Kaggle

Use `notebooks/kaggle_first_real_training.ipynb` for the guided one-cell workflow.
It streams up to 1,500 accepted articles for each language from the
`wikimedia/wikipedia` dataset configurations `20231101.ar` and `20231101.en`.
The script records a source manifest per article and builds a shuffled mixed-language
corpus so the validation tail is not English-only.

The initial configuration uses a 4-layer, 128-dimensional Transformer with
a 4,096-piece SentencePiece vocabulary target and 300 optimization steps.
These are baseline engineering settings, not a claim of sufficient pretraining.

After training, review:
- the training and validation loss trend in `training_report.json`;
- the article counts, dataset revision, and attribution manifests;
- generated Arabic and English samples;
- token count, parameter count, elapsed time, and GPU utilization.

Wikimedia source text is declared under CC BY-SA 3.0 and GFDL in the dataset card.
Keep the manifests with every checkpoint and review ShareAlike/GFDL obligations
before distributing any corpus-derived model weights. See `DATA_POLICY.md`.
