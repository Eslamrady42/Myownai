from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from myownai.config import ModelConfig
from myownai.data import TokenBlockDataset, split_tokens
from myownai.model import LanguageModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loss(
    model: LanguageModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index >= max_batches:
            break
        _, loss = model(inputs.to(device), targets.to(device))
        if loss is not None:
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def train_model(
    *,
    model_config: ModelConfig,
    token_ids: torch.Tensor,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    max_steps: int = 100,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    grad_accum_steps: int = 1,
    eval_interval: int = 25,
    validation_fraction: float = 0.05,
    seed: int = 42,
    resume_from: str | Path | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    if min(max_steps, batch_size, grad_accum_steps, eval_interval) < 1:
        raise ValueError("Training steps, batch size, accumulation, and eval interval must be positive")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # Avoid oversubscribing small desktop CPUs by default.
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    train_tokens, val_tokens = split_tokens(token_ids, validation_fraction)
    train_ds = TokenBlockDataset(train_tokens, model_config.max_seq_len)
    val_ds = TokenBlockDataset(val_tokens, model_config.max_seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers,
    )

    model = LanguageModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_target = output / "tokenizer.model"
    if Path(tokenizer_path).resolve() != tokenizer_target.resolve():
        shutil.copy2(tokenizer_path, tokenizer_target)
    model_config.to_json(output / "model_config.json")

    history: list[dict[str, float | int]] = []
    step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        step = int(checkpoint.get("step", 0))
        if step >= max_steps:
            raise ValueError(
                f"Checkpoint is already at step {step}; choose --max-steps greater than that."
            )
        # Continue the cosine schedule against the new total step target.
        scheduler.last_epoch = step
        scheduler._step_count = step + 1
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]

    model.train()
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        pending_batches = 0
        for batch_index, (inputs, targets) in enumerate(train_loader):
            logits, loss = model(inputs.to(device), targets.to(device))
            if loss is None:
                raise RuntimeError("Training loss was not produced")
            (loss / grad_accum_steps).backward()
            pending_batches += 1

            is_accumulation_boundary = pending_batches >= grad_accum_steps
            is_last_batch = batch_index == len(train_loader) - 1
            if not is_accumulation_boundary and not is_last_batch:
                continue

            # Correct the scale when the final accumulation group is incomplete.
            if pending_batches < grad_accum_steps:
                scale = grad_accum_steps / pending_batches
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            pending_batches = 0

            record: dict[str, float | int] = {
                "step": step,
                "train_loss": float(loss.detach().item()),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate_loss(model, val_loader, device)
                record["validation_loss"] = val_loss
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "model_config": model_config.__dict__,
                        "step": step,
                        "train_loss": record["train_loss"],
                        "validation_loss": val_loss,
                        "seed": seed,
                    },
                    output / "checkpoint.pt",
                )
            history.append(record)
            if step >= max_steps:
                break

    report = {
        "status": "training_completed",
        "device": str(device),
        "parameter_count": model.parameter_count(),
        "steps": step,
        "train_token_count": int(train_tokens.numel()),
        "validation_token_count": int(val_tokens.numel()),
        "history": history,
        "checkpoint": str(output / "checkpoint.pt"),
        "resumed_from": str(resume_from) if resume_from is not None else None,
        "warning": "A completed run does not establish useful language ability; inspect loss and generated samples.",
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
