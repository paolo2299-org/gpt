from __future__ import annotations

import argparse
import re
from pathlib import Path


CHAPTER_RE = re.compile(r"^chapter\s+([ivxlcdm]+|\d+)\.?\]?$", re.IGNORECASE)
STRUCTURAL_RE = re.compile(
    r"^(chapter\s+([ivxlcdm]+|\d+)\.?|volume\s+([ivxlcdm]+|\d+)|end of the .+ volume)$",
    re.IGNORECASE,
)


def normalize_layout_directive(line: str) -> str:
    """Remove Project Gutenberg HTML-conversion layout markers from letter text."""
    line = re.sub(r"^\s*/\*\s*(?:NIND|RIGHT)?\s*", "", line)
    line = re.sub(r"\s*\*/\s*$", "", line)
    return line


def chapter_heading_from_illustration(block: list[str]) -> str | None:
    for raw_line in block:
        line = raw_line.strip()
        match = CHAPTER_RE.match(line)
        if match:
            return f"CHAPTER {match.group(1).upper()}."
    return None


def remove_artifact_lines(text: str) -> list[str]:
    text = text.replace("\ufeff", "\n\n")
    lines = text.splitlines()
    cleaned: list[str] = []
    illustration_block: list[str] | None = None

    for raw_line in lines:
        line = normalize_layout_directive(raw_line.rstrip())
        stripped = line.strip()

        if illustration_block is not None:
            illustration_block.append(stripped)
            if "]" in stripped:
                heading = chapter_heading_from_illustration(illustration_block)
                if heading is not None:
                    cleaned.append("")
                    cleaned.append(heading)
                    cleaned.append("")
                illustration_block = None
            continue

        if stripped.startswith("[Illustration"):
            illustration_block = [stripped]
            if "]" in stripped:
                heading = chapter_heading_from_illustration(illustration_block)
                if heading is not None:
                    cleaned.append("")
                    cleaned.append(heading)
                    cleaned.append("")
                illustration_block = None
            continue

        cleaned.append(line)

    if illustration_block is not None:
        heading = chapter_heading_from_illustration(illustration_block)
        if heading is not None:
            cleaned.append("")
            cleaned.append(heading)
            cleaned.append("")

    return cleaned


def unwrap_paragraph(lines: list[str]) -> str:
    pieces = [line.strip() for line in lines]
    if len(pieces) == 1:
        return pieces[0]
    return " ".join(piece for piece in pieces if piece)


def unwrap_paragraphs(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        paragraphs.append(unwrap_paragraph(current))
        current = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue

        if STRUCTURAL_RE.match(stripped):
            flush()
            paragraphs.append(stripped.upper() if stripped.lower().startswith("chapter") else stripped)
            continue

        current.append(line)

    flush()
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph).strip() + "\n"


def clean_text(text: str) -> str:
    return unwrap_paragraphs(remove_artifact_lines(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean the six-novel Jane Austen corpus.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("texts/jane-austen.txt"),
        help="Source corpus to clean.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("texts/jane-austen.cleaned.txt"),
        help="Cleaned corpus destination.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser()
    destination = args.output.expanduser()

    text = source.read_text(encoding="utf-8")
    cleaned = clean_text(text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(cleaned, encoding="utf-8")

    print(f"Read: {source}")
    print(f"Wrote: {destination}")
    print(f"Characters: {len(text):,} -> {len(cleaned):,}")


if __name__ == "__main__":
    main()
