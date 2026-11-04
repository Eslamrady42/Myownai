import torch
from myownai.config import ModelConfig
from myownai.model import LanguageModel

def tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                       max_seq_len=16, dropout=0.0)

def test_forward_shape_and_loss() -> None:
    model = LanguageModel(tiny_config())
    inputs = torch.randint(0, 64, (2, 12))
    targets = torch.randint(0, 64, (2, 12))
    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 12, 64)
    assert loss is not None and torch.isfinite(loss)

def test_backward_creates_gradients() -> None:
    model = LanguageModel(tiny_config())
    inputs = torch.randint(0, 64, (2, 12))
    targets = torch.randint(0, 64, (2, 12))
    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()

def test_context_limit_is_enforced() -> None:
    model = LanguageModel(tiny_config())
    inputs = torch.randint(0, 64, (1, 17))
    try:
        model(inputs)
    except ValueError as exc:
        assert "max_seq_len" in str(exc)
    else:
        raise AssertionError("Expected context length validation to fail")
