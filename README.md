# My Own AI — Independent Arabic/English Language Model

This is an independent language-model research project. It does **not** call OpenAI, Anthropic, Gemini, or any hosted LLM API.

The first milestone is verifiable: train a tokenizer, initialize a decoder-only Transformer from random weights, train it on a licensed local corpus, save a checkpoint, and run local inference.

> This is a foundation, not a claim of Claude-level capability. Quality depends on data, compute, training time, and evaluation.

## Included
- PyTorch decoder-only Transformer with RoPE, RMSNorm, SwiGLU, and causal attention.
- SentencePiece tokenizer training on your own Arabic/English corpus.
- Local text dataset loader and next-token training.
- Hardware audit utility.
- Checkpointed training and local generation.
- FastAPI inference endpoint loading only this project's checkpoints.
- Responsive Claude-inspired local chat interface.
- Unit tests and GitHub Actions CI.

## Requirements
Python 3.10+, PyTorch 2.2+, and a UTF-8 text corpus you are legally allowed to use. CPU works for tiny experiments; CUDA GPU is strongly recommended for useful training.

## Windows quick start
```powershell
cd my-own-ai
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python scripts/detect_hardware.py
```

## Prepare data and tokenizer
Place licensed UTF-8 `.txt` files in `datasets/corpus/` (subfolders supported). Do not commit private, copyrighted, or unlicensed material. Record every source and license.

For the micro configuration, start with a vocabulary target of 2048:

```powershell
python scripts/train_tokenizer.py --input-dir datasets/corpus --output-dir artifacts/tokenizer --vocab-size 2048
```

Check the printed *actual* vocabulary size. Set `vocab_size` in a copy of `configs/micro.json` to that exact value if it differs.

## First micro training run
```powershell
python scripts/train.py --data-dir datasets/corpus --tokenizer artifacts/tokenizer/tokenizer.model --config configs/micro.json --output-dir artifacts/micro-run --max-steps 100 --batch-size 2
```

Generate locally:
```powershell
python scripts/generate.py --checkpoint artifacts/micro-run/checkpoint.pt --tokenizer artifacts/micro-run/tokenizer.model --prompt "مرحبا، كيف حالك؟" --max-new-tokens 80
```

Run the local web app and API:
```powershell
$env:MYOWNAI_CHECKPOINT = "artifacts/micro-run/checkpoint.pt"
$env:MYOWNAI_TOKENIZER = "artifacts/micro-run/tokenizer.model"
uvicorn myownai.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` for the chat interface or `http://127.0.0.1:8000/docs` for API documentation.

Run tests with `pytest`.

## Limitations
- Initial milestone is text-only.
- No trained weights or datasets are included.
- No commercial AI APIs or pretrained model weights are used.
- The first randomly initialized model will generate poor text; that is expected.
- Memory, RAG, vision, speech, agents, and a full project system are future milestones.
- GitHub stores source code; it does not supply training GPU compute.
- Keep large corpora and checkpoints out of ordinary Git commits.
