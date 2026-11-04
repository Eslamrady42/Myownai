from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from myownai.inference import LocalGenerator

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from your own checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    model = LocalGenerator(args.checkpoint, args.tokenizer)
    print(model.generate(args.prompt, args.max_new_tokens, args.temperature, args.top_p))

if __name__ == "__main__":
    main()
