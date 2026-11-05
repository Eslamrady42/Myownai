# Dataset policy

Use only sources legally usable for the intended training purpose. Record source name, canonical URL, license, attribution requirements, version, download date, language, size, and processing steps. Public availability is not the same as permission to train. Do not ingest private data, credentials, or personal records. Keep raw corpora out of Git.

## Initial public corpus candidate: Wikimedia Wikipedia

The first bilingual prototype uses the `wikimedia/wikipedia` Hugging Face dataset,
with the pinned dump configurations `20231101.ar` and `20231101.en`. The dataset
card identifies the source as cleaned Wikipedia dump articles and declares
CC BY-SA 3.0 and GNU Free Documentation License (GFDL). See
https://huggingface.co/datasets/wikimedia/wikipedia and
https://dumps.wikimedia.org/legal.html.

The downloader streams a bounded number of articles per language, applies Unicode
NFKC and whitespace normalization, basic script-ratio filtering, truncates long
articles, and records per-article title, source URL, dataset revision, license,
processing note, and SHA-256 in JSONL manifests. It does not copy corpus text into
Git. The first run is a small research baseline, not a general-purpose web corpus.

**License caution:** preserve article-level attribution and comply with the
applicable ShareAlike/GFDL conditions. The project does not redistribute raw
Wikipedia text. Do not publish or redistribute checkpoints trained on this corpus
until the licensing implications for model weights and the intended distribution
have been reviewed. This is a dataset engineering notice, not legal advice.
