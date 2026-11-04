from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from myownai.config import ModelConfig
from myownai.data import encode_corpus
from myownai.tokenizer import OwnTokenizer
from myownai.training import train_model

def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the independent language model")
    parser.add_argument("--data-dir", default="datasets/corpus")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", default="configs/micro.json")
    parser.add_argument("--output-dir", default="artifacts/training-run")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()
    config = ModelConfig.from_json(args.config)
    tokenizer = OwnTokenizer(args.tokenizer)
    if tokenizer.vocab_size != config.vocab_size:
        raise SystemExit(
            f"Tokenizer vocabulary ({tokenizer.vocab_size}) does not match "
            f"model vocab_size ({config.vocab_size}). Update the config."
        )
    token_ids = encode_corpus(args.data_dir, tokenizer, max_docs=args.max_docs)
    report = train_model(
        model_config=config, token_ids=token_ids, tokenizer_path=args.tokenizer,
        output_dir=args.output_dir, max_steps=args.max_steps,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
    )
    print(f"Training completed. Report: {Path(args.output_dir) / 'training_report.json'}")
    print(f"Device: {report['device']}; parameters: {report['parameter_count']}")

if __name__ == "__main__":
    main()
