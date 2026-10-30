"""Load markdown documents and split them into heading-aware chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a markdown file into (section_heading, body) pairs on '## ' headings."""
    sections: list[tuple[str, str]] = []
    current_heading = "Overview"
    buffer: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if "".join(buffer).strip():
                sections.append((current_heading, "\n".join(buffer).strip()))
            current_heading = line[3:].strip()
            buffer = []
        elif line.startswith("# "):
            continue  # document title is stored separately
        else:
            buffer.append(line)
    if "".join(buffer).strip():
        sections.append((current_heading, "\n".join(buffer).strip()))
    return sections


def _window(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = size - overlap
    return [words[i : i + size] for i in range(0, len(words) - overlap, step)]


def chunk_document(path: Path, max_words: int = 120, overlap: int = 20) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", raw, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem
    doc_id = path.stem
    chunks: list[Chunk] = []
    for s_idx, (heading, body) in enumerate(_split_sections(raw)):
        for w_idx, words in enumerate(_window(body.split(), max_words, overlap)):
            text = " ".join(words)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{s_idx}.{w_idx}",
                    doc_id=doc_id,
                    title=title,
                    section=heading,
                    # Prefix title + section so the chunk keeps its context when retrieved alone.
                    text=f"{title} — {heading}: {text}",
                    metadata={"source": str(path.name)},
                )
            )
    return chunks


def load_corpus(folder: str | Path) -> list[Chunk]:
    folder = Path(folder)
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        chunks.extend(chunk_document(path))
    if not chunks:
        raise ValueError(f"No markdown documents found in {folder}")
    return chunks
