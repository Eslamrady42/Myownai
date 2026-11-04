from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import Dataset
from myownai.tokenizer import OwnTokenizer

def find_text_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return sorted(p for p in root.rglob("*.txt") if p.is_file() and p.stat().st_size > 0)

def encode_corpus(data_dir: str | Path, tokenizer: OwnTokenizer, max_docs: int | None = None) -> torch.Tensor:
    files = find_text_files(data_dir)
    if max_docs is not None:
        files = files[:max_docs]
    if not files:
        raise ValueError(f"No non-empty .txt files found under {data_dir}")
    ids: list[int] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="strict")
        if text.strip():
            ids.extend(tokenizer.encode(text, add_eos=True))
    if len(ids) < 3:
        raise ValueError("Corpus is too small; provide more text before training")
    return torch.tensor(ids, dtype=torch.long)

class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, token_ids: torch.Tensor, sequence_length: int) -> None:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if token_ids.numel() < sequence_length + 1:
            raise ValueError(f"Need at least {sequence_length + 1} tokens, got {token_ids.numel()}")
        self.tokens = token_ids
        self.sequence_length = sequence_length
        self.blocks = (token_ids.numel() - 1) // sequence_length

    def __len__(self) -> int:
        return self.blocks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        chunk = self.tokens[start:start + self.sequence_length + 1]
        return chunk[:-1], chunk[1:]

def split_tokens(token_ids: torch.Tensor, validation_fraction: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    split_at = int(token_ids.numel() * (1.0 - validation_fraction))
    if split_at < 2 or token_ids.numel() - split_at < 2:
        raise ValueError("Corpus is too small for a train/validation split")
    return token_ids[:split_at], token_ids[split_at:]
