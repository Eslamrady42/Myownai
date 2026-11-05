from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

# Kaggle images differ; ensure runtime-only data dependencies exist before
# importing modules that depend on SentencePiece.
def ensure_package(import_name: str, package_spec: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", package_spec],
            check=True,
        )


ensure_package("sentencepiece", "sentencepiece>=0.2")
ensure_package("datasets", "datasets>=3.3")
ensure_package("huggingface_hub", "huggingface_hub>=0.27")

from myownai.config import ModelConfig
from myownai.data import encode_corpus
from myownai.inference import LocalGenerator
from myownai.tokenizer import OwnTokenizer
from myownai.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a small bilingual Wikipedia corpus and train a from-scratch micro-model."
    )
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--articles-per-language", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-chars", type=int, default=250)
    parser.add_argument("--max-chars", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if min(args.articles_per_language, args.steps, args.batch_size, args.vocab_size) < 1:
        parser.error("article count, steps, batch size and vocabulary size must be positive")

    has_cuda = torch.cuda.is_available()
    if args.require_gpu and not has_cuda:
        raise RuntimeError(
            "GPU is required but not visible. Select a GPU accelerator in Kaggle settings and restart."
        )
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    print(f"Training device: {device_name}", flush=True)

    work_root = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else ROOT / "artifacts"
    work_root.mkdir(parents=True, exist_ok=True)
    corpus_dir = work_root / "myownai_wikipedia_corpus"
    artifacts_dir = work_root / "myownai_wikipedia_micro_run"
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Downloading a balanced Arabic/English corpus ===", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_wikipedia_corpus.py"),
            "--output-dir", str(corpus_dir),
            "--articles-per-language", str(args.articles_per_language),
            "--min-chars", str(args.min_chars),
            "--max-chars", str(args.max_chars),
            "--seed", str(args.seed),
            "--languages", "ar", "en",
        ],
        cwd=ROOT,
        check=True,
    )

    mixed_path = corpus_dir / "mixed.txt"
    if not mixed_path.is_file() or mixed_path.stat().st_size == 0:
        raise RuntimeError("The bilingual corpus file was not created.")

    # Encode only the shuffled bilingual file; do not double-count ar.txt/en.txt.
    training_input = work_root / "myownai_training_input"
    if training_input.exists():
        shutil.rmtree(training_input)
    training_input.mkdir(parents=True)
    shutil.copy2(mixed_path, training_input / "mixed.txt")

    print("\n=== Training the tokenizer from random initialization ===", flush=True)
    tokenizer_path = OwnTokenizer.train(
        [training_input / "mixed.txt"],
        artifacts_dir / "tokenizer",
        vocab_size=args.vocab_size,
    )
    tokenizer = OwnTokenizer(tokenizer_path)
    print(f"Actual vocabulary size: {tokenizer.vocab_size}", flush=True)

    print("\n=== Preparing model and tokens ===", flush=True)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        max_seq_len=256,
        dropout=0.1,
    )
    token_ids = encode_corpus(training_input, tokenizer)
    print(f"Training tokens: {token_ids.numel():,}", flush=True)

    print("\n=== Training from random weights ===", flush=True)
    report = train_model(
        model_config=config,
        token_ids=token_ids,
        tokenizer_path=tokenizer_path,
        output_dir=artifacts_dir,
        max_steps=args.steps,
        batch_size=args.batch_size if has_cuda else min(args.batch_size, 2),
        grad_accum_steps=1,
        learning_rate=args.learning_rate,
        eval_interval=max(1, args.steps // 10),
        validation_fraction=0.05,
        seed=args.seed,
    )

    for source_name in ("corpus_metadata.json", "ar.jsonl", "en.jsonl"):
        source = corpus_dir / source_name
        if source.is_file():
            shutil.copy2(source, artifacts_dir / source_name)

    print("\n=== Checking Arabic and English generation ===", flush=True)
    generator = LocalGenerator(
        artifacts_dir / "checkpoint.pt",
        artifacts_dir / "tokenizer.model",
    )
    generations = {
        "arabic_prompt": "ما هي أهمية العلم في حياة الإنسان؟",
        "arabic_output": generator.generate("ما هي أهمية العلم في حياة الإنسان؟", max_new_tokens=80),
        "english_prompt": "Why is learning important?",
        "english_output": generator.generate("Why is learning important?", max_new_tokens=80),
    }
    for key, value in generations.items():
        print(f"{key}: {value}", flush=True)

    metadata = json.loads((corpus_dir / "corpus_metadata.json").read_text(encoding="utf-8"))
    summary = {
        "status": "training_completed",
        "device": report["device"],
        "gpu_name": device_name,
        "parameters": report["parameter_count"],
        "steps": report["steps"],
        "training_tokens": report["train_token_count"],
        "validation_tokens": report["validation_token_count"],
        "final_train_loss": report["history"][-1]["train_loss"],
        "final_validation_loss": report["history"][-1].get("validation_loss"),
        "checkpoint": str(artifacts_dir / "checkpoint.pt"),
        "corpus_metadata": metadata,
        "generations": generations,
        "warning": (
            "This is an initial small from-scratch pretraining run, not an assistant-level model. "
            "Do not distribute corpus-derived weights until licensing obligations are reviewed."
        ),
    }
    (artifacts_dir / "real_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    archive = shutil.make_archive(
        str(work_root / "Myownai_Wikipedia_Micro_Run"),
        "zip",
        artifacts_dir,
    )
    print("\n=== FINAL RESULT ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"\n✅ Real-corpus training run finished. Results archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
