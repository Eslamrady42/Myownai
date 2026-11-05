from pathlib import Path

from myownai.data import encode_corpus


class RecordingTokenizer:
    def __init__(self) -> None:
        self.documents: list[str] = []

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        self.documents.append(text)
        return [len(self.documents), 99] if add_eos else [len(self.documents)]


def test_encode_corpus_adds_eos_per_blank_line_document(tmp_path: Path) -> None:
    corpus = tmp_path / "sample.txt"
    corpus.write_text("First document.\n\nSecond document.", encoding="utf-8")
    tokenizer = RecordingTokenizer()

    ids = encode_corpus(tmp_path, tokenizer)  # type: ignore[arg-type]

    assert tokenizer.documents == ["First document.", "Second document."]
    assert ids.tolist() == [1, 99, 2, 99]
