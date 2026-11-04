import pytest
from myownai.config import ModelConfig

def test_config_rejects_odd_rope_head_dimension() -> None:
    with pytest.raises(ValueError, match="even"):
        ModelConfig(vocab_size=64, d_model=30, n_layers=2, n_heads=6)

def test_config_accepts_valid_values() -> None:
    config = ModelConfig(vocab_size=128, d_model=32, n_layers=2, n_heads=4, max_seq_len=32)
    assert config.d_model // config.n_heads == 8
