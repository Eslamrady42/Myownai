from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset
from huggingface_hub import HfApi

from myownai.corpus import clean_article_text


DATASET_ID = "wikimedia/wikipedia"
DATASET_CONFIGS = {"ar": "20231101.ar", "en": "20231101.en"}
DATASET_LICENSE = "CC BY-SA 3.0 and GNU Free Documentation License (GFDL)"
DATASET_CARD_URL = "https://huggingface.co/datasets/wikimedia/wikipedia"
LICENSE_URL = "https://dumps.wikimedia.org/legal.html"


def resolve_dataset_revision() -> str:
    try:
        return HfApi().dataset_info(DATASET_ID).sha
    except Exception:
        # Load still works with main; report that the exact revision could not be resolved.
        return "main (revision lookup unavailable)"


def collect_language(
    language: str,
    output_dir: Path,
    articles_per_language: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    revision: str,
) -> dict[str, Any]:
    config = DATASET_CONFIGS[language]
    print(f"Loading {DATASET_ID}:{config} (streaming, revision={revision})", flush=True)
    stream = load_dataset(
        DATASET_ID,
        config,
        split="train",
        streaming=True,
        revision=revision if not revision.endswith("(revision lookup unavailable)") else "main",
    )
    stream = stream.shuffle(seed=seed, buffer_size=max(1000, min(10000, articles_per_language * 2)))

    text_path = output_dir / f"{language}.txt"
    manifest_path = output_dir / f"{language}.jsonl"
    seen: set[str] = set()
    accepted = 0
    examined = 0
    rejected = 0

    with text_path.open("w", encoding="utf-8", newline="\n") as text_file, manifest_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as manifest_file:
        for row in stream:
            examined += 1
            article = clean_article_text(
                str(row.get("text") or ""),
                language=language,
                min_chars=min_chars,
                max_chars=max_chars,
            )
            if article is None:
                rejected += 1
                continue
            digest = hashlib.sha256(article.encode("utf-8")).hexdigest()
            if digest in seen:
                rejected += 1
                continue
            seen.add(digest)

            title = str(row.get("title") or "").strip()
            source_url = str(row.get("url") or "").strip()
            text_file.write(article.replace("\n", " ") + "\n\n")
            manifest_file.write(
                json.dumps(
                    {
                        "language": language,
                        "title": title,
                        "url": source_url,
                        "source_dataset": DATASET_ID,
                        "source_config": config,
                        "source_revision": revision,
                        "license": DATASET_LICENSE,
                        "license_url": LICENSE_URL,
                        "processing": "NFKC normalization, whitespace normalization, script-ratio filter, truncation to max_chars",
                        "sha256": digest,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            accepted += 1
            if accepted % 100 == 0:
                print(
                    f"{language}: accepted={accepted}/{articles_per_language}, "
                    f"examined={examined}, rejected={rejected}",
                    flush=True,
                )
            if accepted >= articles_per_language:
                break

    if accepted == 0:
        raise RuntimeError(
            f"No {language} articles passed filters. Check dataset access or lower --min-chars."
        )
    return {
        "language": language,
        "config": config,
        "articles": accepted,
        "examined": examined,
        "rejected": rejected,
        "text_file": str(text_path),
        "manifest_file": str(manifest_path),
        "text_bytes": text_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a balanced, licensed Arabic/English Wikipedia corpus."
    )
    parser.add_argument("--output-dir", default="/kaggle/working/myownai_wikipedia_corpus")
    parser.add_argument("--articles-per-language", type=int, default=1500)
    parser.add_argument("--min-chars", type=int, default=250)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--languages", nargs="+", choices=["ar", "en"], default=["ar", "en"])
    args = parser.parse_args()

    if args.articles_per_language < 1:
        parser.error("--articles-per-language must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = args.revision
    if revision == "main":
        print("Resolving the current public dataset revision...", flush=True)
        try:
            revision = resolve_dataset_revision()
        except Exception:
            revision = "main"
    print(f"Resolved revision: {revision}", flush=True)

    reports = []
    for language in args.languages:
        reports.append(
            collect_language(
                language=language,
                output_dir=output_dir,
                articles_per_language=args.articles_per_language,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                seed=args.seed,
                revision=revision,
            )
        )

    documents: list[str] = []
    for language in args.languages:
        language_text = (output_dir / f"{language}.txt").read_text(encoding="utf-8")
        documents.extend(doc.strip() for doc in language_text.split("\n\n") if doc.strip())
    random.Random(args.seed).shuffle(documents)
    mixed_path = output_dir / "mixed.txt"
    mixed_path.write_text("\n\n".join(documents) + "\n", encoding="utf-8")

    metadata = {
        "status": "prepared",
        "dataset": DATASET_ID,
        "dataset_card": DATASET_CARD_URL,
        "dataset_revision": revision,
        "dataset_license": DATASET_LICENSE,
        "license_url": LICENSE_URL,
        "license_notice": (
            "Wikimedia Wikipedia text is declared under CC BY-SA 3.0 and GFDL in the "
            "dataset card. Preserve attribution and comply with ShareAlike/GFDL terms. "
            "Do not redistribute raw corpus or model weights until the applicable "
            "obligations have been reviewed."
        ),
        "date_prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "mixed_corpus": str(mixed_path),
        "mixed_articles": len(documents),
        "reports": reports,
    }
    (output_dir / "corpus_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nCorpus prepared:", flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
