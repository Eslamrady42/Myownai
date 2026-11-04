from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from myownai.data import find_text_files
from myownai.tokenizer import OwnTokenizer

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the project's SentencePiece tokenizer")
    parser.add_argument("--input-dir", default="datasets/corpus")
    parser.add_argument("--output-dir", default="artifacts/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8000)
    args = parser.parse_args()
    files = find_text_files(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = OwnTokenizer.train(files, output_dir / "tokenizer", args.vocab_size)
    tokenizer = OwnTokenizer(model_path)
    print(f"Tokenizer saved to: {model_path}")
    print(f"Actual vocabulary size: {tokenizer.vocab_size}")

if __name__ == "__main__":
    main()
