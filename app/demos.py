"""Registry of demos served by the web app.

A demo pairs a model checkpoint (a preset + a weights file) with the task it
illustrates and the copy shown around it. The set of demos is static; which ones
are *available* depends on whether their weights file is present in
``WEIGHTS_DIR`` (so a deployment shows only the models it actually ships).

``task`` is the seam for future chapters: today only ``"completion"`` is wired
up; ch6 will add ``"classification"`` and ch7 ``"instruction"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_BASE_URL = "https://github.com/paolo2299-org/gpt/tree/main"


@dataclass(frozen=True)
class Demo:
    slug: str  # URL segment, e.g. "jane", "gpt2"
    title: str
    tagline: str  # one-liner for the landing card
    description: str  # paragraph shown on the demo page
    task: str  # "completion" (future: "classification", "instruction")
    preset: str  # key into llm.config.MODEL_PRESETS
    weights_filename: str  # file inside WEIGHTS_DIR
    examples: tuple[str, ...] = ()
    author_name: str = ""
    repo_subpath: str = "llm"  # deep link for "see the code"

    @property
    def repo_url(self) -> str:
        return f"{REPO_BASE_URL}/{self.repo_subpath}"


DEMOS: tuple[Demo, ...] = (
    Demo(
        slug="jane",
        title="JaneGPT",
        tagline="GPT-2 small fine-tuned on the six major Jane Austen novels.",
        description=(
            "Start with OpenAI's GPT-2 small checkpoint, then fine-tune it further "
            "on the six major Jane Austen novels. The result keeps GPT-2's general "
            "language ability while nudging completions toward Austen's diction, "
            "social comedy, and long-form sentence rhythm. Start a phrase and watch "
            "the model continue it in a more Austen-flavoured voice."
        ),
        task="completion",
        preset="gpt2-small",
        weights_filename="jane-austen-gpt2-small.best.pth",
        examples=("It is a truth", "She had long suspected", "The visit was", "In the drawing-room"),
        author_name="Jane Austen + GPT-2",
        repo_subpath="scripts/finetune_llm.py",
    ),
    Demo(
        slug="gpt2",
        title="GPT-2 (124M)",
        tagline="OpenAI's original GPT-2 small, loaded from its open weights.",
        description=(
            "OpenAI's publicly released GPT-2 (124M) weights loaded into the same "
            "GPT architecture used throughout these demos — exactly as covered in "
            "chapter 5 of the book. Because it was trained on a large slice of the "
            "web, its completions are broad and fluent. Start a phrase and watch it "
            "continue."
        ),
        task="completion",
        preset="gpt2-small",
        weights_filename="gpt2-small.pth",
        examples=(
            "Every effort moves you",
            "Once upon a time",
            "The meaning of life is",
            "In a shocking finding, scientists discovered",
        ),
        author_name="OpenAI GPT-2",
        repo_subpath="scripts/fetch_gpt2.py",
    ),
    Demo(
        slug="gpt2-medium",
        title="GPT-2 (355M)",
        tagline="OpenAI's larger GPT-2 (355M), loaded from its open weights.",
        description=(
            "The same architecture as the other demos, scaled up to OpenAI's "
            "355M-parameter GPT-2 (24 layers, 1024-dim embeddings) and loaded from "
            "the released open weights — book chapter 5. Larger than the 124M model, "
            "so its completions tend to hang together better. Start a phrase and "
            "watch it continue."
        ),
        task="completion",
        preset="gpt2-medium",
        weights_filename="gpt2-medium.pth",
        examples=(
            "Every effort moves you",
            "Once upon a time",
            "The meaning of life is",
            "In a shocking finding, scientists discovered",
        ),
        author_name="OpenAI GPT-2",
        repo_subpath="scripts/fetch_gpt2.py",
    ),
)


def get_demo(slug: str) -> Demo | None:
    return next((demo for demo in DEMOS if demo.slug == slug), None)
