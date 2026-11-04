from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from myownai.inference import LocalGenerator

app = FastAPI(title="My Own AI — Local Inference", version="0.1.0",
              description="Runs only project-trained checkpoints; no hosted LLM APIs.")

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    max_new_tokens: int = Field(default=80, ge=1, le=1024)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)

@lru_cache(maxsize=1)
def load_generator() -> LocalGenerator:
    checkpoint = os.getenv("MYOWNAI_CHECKPOINT", "")
    tokenizer = os.getenv("MYOWNAI_TOKENIZER", "")
    if not checkpoint or not tokenizer:
        raise RuntimeError("Set MYOWNAI_CHECKPOINT and MYOWNAI_TOKENIZER")
    return LocalGenerator(checkpoint, tokenizer)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "my-own-ai-local-inference"}

@app.get("/model/status")
def model_status() -> dict[str, object]:
    try:
        generator = load_generator()
        return {"loaded": True, "device": str(generator.device),
                "training_step": generator.step,
                "parameters": generator.model.parameter_count(),
                "vocab_size": generator.config.vocab_size,
                "max_seq_len": generator.config.max_seq_len}
    except Exception as exc:
        return {"loaded": False, "reason": str(exc)}

@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, str]:
    try:
        generator = load_generator()
        text = generator.generate(request.prompt, max_new_tokens=request.max_new_tokens,
                                  temperature=request.temperature, top_p=request.top_p)
        return {"text": text, "model": "my-own-ai"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

# Serve the same-origin web UI when the project is run as a local application.
frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
