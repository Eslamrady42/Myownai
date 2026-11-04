from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 2048
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 256
    dropout: float = 0.1
    bias: bool = False
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.vocab_size < 8:
            raise ValueError("vocab_size must be at least 8")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("d_model, n_layers, and n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def to_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
