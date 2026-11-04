from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sentencepiece as spm

class OwnTokenizer:
    """SentencePiece tokenizer trained from this project's own corpus."""
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Tokenizer model not found: {self.model_path}")
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))

    @classmethod
    def train(cls, input_files: Iterable[str | Path], model_prefix: str | Path, vocab_size: int = 8000) -> Path:
        files = [Path(item) for item in input_files]
        files = [item for item in files if item.is_file() and item.stat().st_size > 0]
        if not files:
            raise ValueError("No non-empty UTF-8 training text files were found")
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        prefix = Path(model_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(input=','.join(str(p.resolve()) for p in files), model_prefix=str(prefix), model_type='bpe', vocab_size=vocab_size, character_coverage=0.9995, pad_id=0, unk_id=1, bos_id=2, eos_id=3, hard_vocab_limit=False, shuffle_input_sentence=True)
        return prefix.with_suffix('.model')

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        ids = self.processor.encode(text, out_type=int)
        if add_bos:
            ids.insert(0, self.processor.bos_id())
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return self.processor.decode(token_ids)
