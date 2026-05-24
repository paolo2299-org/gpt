"""Completion page routes."""

from __future__ import annotations

from dataclasses import dataclass

from flask import Blueprint, current_app, render_template, request

bp = Blueprint("pages", __name__)


@dataclass
class CompletionForm:
    prompt: str = ""


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
                    max_new_tokens=current_app.config["DEFAULT_MAX_NEW_TOKENS"],
                    temperature=current_app.config["DEFAULT_TEMPERATURE"],
                    top_k=current_app.config["DEFAULT_TOP_K"],
                    seed=current_app.config["DEFAULT_SEED"],
                )

    return render_template(
        "index.html",
        form=form,
        completion=completion,
        error=error,
        model_name=current_app.config["MODEL_PRESET"],
        weights_path=current_app.config["MODEL_WEIGHTS_PATH"],
    )


def _form_from_request() -> CompletionForm:
    if request.method == "POST":
        return CompletionForm(prompt=request.form.get("prompt", ""))
    return CompletionForm()
