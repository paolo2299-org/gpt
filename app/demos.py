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
        tagline="A homemade LLM trained from scratch on the works of Jane Austen.",
        description=(
            "A homemade LLM trained solely on the works of Jane Austen. "
            "Start a phrase and generate text that Jane Austen might have written. "
            "As you will see, it doesn't live up to the real Jane Austen, or even "
            "make a whole lot of sense, but it does have the flavour of the original "
            "texts. Realistic results could be achieved with this model, but it would "
            "take terabytes of data and millions of dollars in GPU costs."
        ),
        task="completion",
        preset="124M",
        weights_filename="model.dickens.pth",
        examples=("It was a truth", "The morning was", "She had never", "In the drawing-room"),
        author_name="Jane Austen",
        repo_subpath="llm",
    ),
    Demo(
        slug="gpt2",
        title="GPT-2 (124M)",
        tagline="OpenAI's original GPT-2 small, loaded from its open weights.",
        description=(
            "The same GPT architecture as JaneGPT, but with OpenAI's publicly "
            "released GPT-2 (124M) weights loaded in — exactly as covered in "
            "chapter 5 of the book. Because it was trained on a large slice of the "
            "web, its completions are far more fluent than the from-scratch demo. "
            "Start a phrase and watch it continue."
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
)


def get_demo(slug: str) -> Demo | None:
    return next((demo for demo in DEMOS if demo.slug == slug), None)
