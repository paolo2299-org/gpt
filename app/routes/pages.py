"""Completion page routes."""

from __future__ import annotations

from dataclasses import dataclass

from flask import Blueprint, current_app, render_template, request

bp = Blueprint("pages", __name__)


@dataclass
class CompletionForm:
    prompt: str = ""
    max_new_tokens: int = 80
    temperature: float = 0.0
    top_k: int | None = 50
    seed: int = 123


@bp.route("/", methods=["GET", "POST"])
def index() -> str:
    form = _form_from_request()
    completion = ""
    error = ""

    if request.method == "POST":
        if not form.prompt.strip():
            error = "Enter a prompt to continue."
        else:
            completer = current_app.extensions.get("llm_completer")
            if completer is None:
                error = "The model is not loaded."
            else:
                completion = completer.complete(
                    prompt=form.prompt,
                    max_new_tokens=form.max_new_tokens,
                    temperature=form.temperature,
                    top_k=form.top_k,
                    seed=form.seed,
                )

    return render_template(
        "index.html",
        form=form,
        completion=completion,
        error=error,
        model_name=current_app.config["MODEL_PRESET"],
        weights_path=current_app.config["MODEL_WEIGHTS_PATH"],
        max_new_tokens_limit=current_app.config["MAX_NEW_TOKENS_LIMIT"],
    )


def _form_from_request() -> CompletionForm:
    defaults = current_app.config
    if request.method != "POST":
        return CompletionForm(
            max_new_tokens=defaults["DEFAULT_MAX_NEW_TOKENS"],
            temperature=defaults["DEFAULT_TEMPERATURE"],
            top_k=defaults["DEFAULT_TOP_K"],
            seed=defaults["DEFAULT_SEED"],
        )

    return CompletionForm(
        prompt=request.form.get("prompt", ""),
        max_new_tokens=_bounded_int(
            request.form.get("max_new_tokens"),
            default=defaults["DEFAULT_MAX_NEW_TOKENS"],
            minimum=1,
            maximum=defaults["MAX_NEW_TOKENS_LIMIT"],
        ),
        temperature=_bounded_float(
            request.form.get("temperature"),
            default=defaults["DEFAULT_TEMPERATURE"],
            minimum=0.0,
            maximum=2.0,
        ),
        top_k=_optional_bounded_int(
            request.form.get("top_k"),
            minimum=1,
            maximum=200,
        ),
        seed=_bounded_int(
            request.form.get("seed"),
            default=defaults["DEFAULT_SEED"],
            minimum=0,
            maximum=2**31 - 1,
        ),
    )


def _bounded_int(value: str | None, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except ValueError:
        parsed = default
    return min(max(parsed, minimum), maximum)


def _optional_bounded_int(value: str | None, minimum: int, maximum: int) -> int | None:
    if value is None or value.strip() == "":
        return None
    return _bounded_int(value, default=minimum, minimum=minimum, maximum=maximum)


def _bounded_float(value: str | None, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except ValueError:
        parsed = default
    return min(max(parsed, minimum), maximum)

