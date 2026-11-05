from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tiny end-to-end Myownai training smoke test on Kaggle."
    )
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    if not (source_root / "myownai" / "__init__.py").is_file():
        raise FileNotFoundError(f"Myownai source package not found under {source_root}")
    sys.path.insert(0, str(source_root))

    print("=== Runtime ===", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}", flush=True)
    if args.require_gpu and not has_cuda:
        raise RuntimeError(
            "GPU was required but CUDA is unavailable. Select a GPU accelerator in Kaggle "
            "Notebook settings, restart the session, and rerun."
        )
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    print(f"Device: {device_name}", flush=True)
    if has_cuda:
        print(
            f"VRAM GiB: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}",
            flush=True,
        )

    try:
        import sentencepiece  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "sentencepiece>=0.2"],
            check=True,
        )

    from myownai.config import ModelConfig
    from myownai.data import encode_corpus
    from myownai.inference import LocalGenerator
    from myownai.tokenizer import OwnTokenizer
    from myownai.training import train_model

    work_dir = Path(os.environ.get("MYOWNAI_WORK_DIR", "/kaggle/working/myownai_smoke_test"))
    if work_dir.exists():
        shutil.rmtree(work_dir)
    corpus_dir = work_dir / "corpus"
    artifacts_dir = work_dir / "artifacts"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Synthetic smoke-test corpus ===", flush=True)
    samples = [
        "السلام عليكم، هذا اختبار تقني لنظام التدريب وحفظ النموذج.",
        "نحن نبني نموذج لغة مستقلًا ونتحقق من عمل خطوات التدريب.",
        "هذا النص تجريبي ولا يمثل بيانات لغوية حقيقية أو معرفة عامة.",
        "يجب استخدام بيانات نملك حق استخدامها عند بدء التدريب الحقيقي.",
        "The purpose of this sentence is to test the training pipeline.",
        "We are checking tokenization, model training, checkpoints and generation.",
        "This synthetic corpus is only for a software smoke test.",
        "The model starts with random weights and has not learned language yet.",
    ]
    corpus_path = corpus_dir / "smoke_test.txt"
    corpus_path.write_text(
        "\n".join(
            f"{line} رقم العينة {index} Sample number {index}."
            for index in range(600)
            for line in samples
        ),
        encoding="utf-8",
    )
    print(f"Corpus size: {corpus_path.stat().st_size / 1024:.1f} KiB", flush=True)

    print("\n=== Train tokenizer ===", flush=True)
    tokenizer_path = OwnTokenizer.train(
        [corpus_path], artifacts_dir / "tokenizer", vocab_size=512
    )
    tokenizer = OwnTokenizer(tokenizer_path)
    print(f"Actual vocabulary size: {tokenizer.vocab_size}", flush=True)

    print("\n=== Train tiny random-initialized model ===", flush=True)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=64,
        dropout=0.05,
    )
    token_ids = encode_corpus(corpus_dir, tokenizer)
    report = train_model(
        model_config=config,
        token_ids=token_ids,
        tokenizer_path=tokenizer_path,
        output_dir=artifacts_dir,
        max_steps=args.steps,
        batch_size=8 if has_cuda else 2,
        grad_accum_steps=1,
        eval_interval=max(1, args.steps // 4),
        validation_fraction=0.1,
        seed=42,
    )
    print(f"Training device: {report['device']}", flush=True)
    print(f"Parameters: {report['parameter_count']:,}", flush=True)
    print(f"Steps completed: {report['steps']}", flush=True)
    print(f"Final training loss: {report['history'][-1]['train_loss']:.4f}", flush=True)

    print("\n=== Local generation smoke test ===", flush=True)
    generator = LocalGenerator(
        artifacts_dir / "checkpoint.pt", artifacts_dir / "tokenizer.model"
    )
    sample = generator.generate("السلام عليكم", max_new_tokens=40)
    print(f"Generated sample: {sample}", flush=True)

    required = [
        artifacts_dir / "checkpoint.pt",
        artifacts_dir / "tokenizer.model",
        artifacts_dir / "model_config.json",
        artifacts_dir / "training_report.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Training did not create required artifacts: {missing}")

    archive_base = Path("/kaggle/working/myownai_smoke_test_results")
    archive = Path(shutil.make_archive(str(archive_base), "zip", artifacts_dir))
    result = {
        "status": "passed",
        "device": report["device"],
        "gpu_name": device_name,
        "cuda_available": has_cuda,
        "parameters": report["parameter_count"],
        "steps": report["steps"],
        "final_train_loss": report["history"][-1]["train_loss"],
        "checkpoint": str(artifacts_dir / "checkpoint.pt"),
        "archive": str(archive),
        "note": (
            "Synthetic-data software smoke test only; this is not a trained assistant "
            "or evidence of useful language ability."
        ),
    }
    (work_dir / "smoke_test_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== RESULT ===", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("\n✅ End-to-end smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
