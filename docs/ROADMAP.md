# Training roadmap

## Milestone 0
- Independent decoder-only Transformer, RoPE, RMSNorm, SwiGLU, and causal attention.
- SentencePiece tokenizer training from project-owned/authorized corpus files.
- Local text data pipeline, next-token training, checkpoints, local generation, FastAPI endpoint, and tests.

## Milestone 1
- Run CI and local smoke tests.
- Run the hardware detector on the user's computer.
- Assemble licensed Arabic/English corpus and manifest.
- Complete a real micro-model training run and record its validation loss and generated samples.

## Milestone 2
- Add sharded datasets, resume training, mixed precision, gradient checkpointing, throughput profiling, and evaluation reports.
- Scale only after hardware fit and data volume are measured.

## Milestone 3
- Build supervised instruction tuning with curated, licensed examples and held-out evaluation.
- Evaluate Arabic, English, and mixed-language behavior.

## Milestone 4
- Add local-only chat frontend, conversations, projects, and user-controlled memory.
- Add document retrieval with transparent citations.

## Milestone 5
- Research separately trained vision/audio components and permissioned agent tools.

A finished training script does not imply that the resulting model is useful or comparable to a commercial assistant. Report evidence, not claims.
