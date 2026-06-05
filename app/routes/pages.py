"""Landing page and per-demo routes."""

from __future__ import annotations

from dataclasses import dataclass

from flask import Blueprint, abort, current_app, render_template, request

bp = Blueprint("pages", __name__)


@dataclass
class CompletionForm:
    prompt: str = ""


@bp.route("/")
def index() -> str:
    registry = current_app.extensions["demos"]
    return render_template(
        "landing.html",
        demos=registry.available(),
        site_title=current_app.config["SITE_TITLE"],
    )


@bp.route("/<slug>", methods=["GET", "POST"])
def demo_page(slug: str) -> str:
    registry = current_app.extensions["demos"]
    demo = registry.get(slug)
    if demo is None:
        abort(404)

    if demo.task == "completion":
        return _completion_view(demo, registry)

    # Other tasks (classification, instruction) are not wired up yet.
    abort(404)


def _completion_view(demo, registry) -> str:
    form = _form_from_request()
    completion = ""
    error = ""

    if request.method == "POST":
        prompt = form.prompt.strip()
        if not prompt:
            error = "Enter a prompt to continue."
        else:
            runner = registry.runner_for(demo)
            completion = runner.complete(
                prompt=prompt,
                max_new_tokens=current_app.config["DEFAULT_MAX_NEW_TOKENS"],
                temperature=current_app.config["DEFAULT_TEMPERATURE"],
                top_k=current_app.config["DEFAULT_TOP_K"],
                seed=current_app.config["DEFAULT_SEED"],
            )

    return render_template(
        "completion.html",
        demo=demo,
        form=form,
        completion=completion,
        error=error,
    )


def _form_from_request() -> CompletionForm:
    if request.method == "POST":
        return CompletionForm(prompt=request.form.get("prompt", ""))
    return CompletionForm()
