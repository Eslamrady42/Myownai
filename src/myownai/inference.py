from __future__ import annotations
from pathlib import Path
import torch
from torch.nn import functional as F
from myownai.config import ModelConfig
from myownai.model import LanguageModel
from myownai.tokenizer import OwnTokenizer

class LocalGenerator:
    def __init__(self, checkpoint_path: str | Path, tokenizer_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = ModelConfig(**checkpoint["model_config"])
        self.model = LanguageModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.tokenizer = OwnTokenizer(tokenizer_path)
        self.step = int(checkpoint.get("step", 0))

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 80, temperature: float = 0.8, top_p: float = 0.9) -> str:
        if max_new_tokens < 1 or temperature <= 0 or not 0.0 < top_p <= 1.0:
            raise ValueError("Invalid generation settings")
        ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        if not ids:
            ids = [self.tokenizer.processor.bos_id()]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self.model(context)
            scores = logits[:, -1, :] / temperature
            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative - sorted_probs > top_p
            sorted_logits[remove] = -torch.inf
            filtered = torch.full_like(scores, -torch.inf)
            filtered.scatter_(1, sorted_indices, sorted_logits)
            next_id = torch.multinomial(F.softmax(filtered, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            if int(next_id.item()) == self.tokenizer.eos_id:
                break
        return self.tokenizer.decode(input_ids[0, len(ids):].tolist())
