from __future__ import annotations

from app.services.completer import trim_to_complete_sentence


def test_trim_to_complete_sentence_removes_incomplete_tail():
    text = "She had long suspected that he was right. But she could not"

    assert trim_to_complete_sentence(text, prompt="She had long suspected") == (
        "She had long suspected that he was right."
    )


def test_trim_to_complete_sentence_ignores_common_title_abbreviations():
    text = "She greeted Mr. Darcy and Mrs. Bennet with composure. The morning was"

    assert trim_to_complete_sentence(text, prompt="She greeted") == (
        "She greeted Mr. Darcy and Mrs. Bennet with composure."
    )


def test_trim_to_complete_sentence_keeps_closing_quote_after_period():
    text = "It was impossible to reply, “I am quite convinced.” But the"

    assert trim_to_complete_sentence(text, prompt="It was impossible") == (
        "It was impossible to reply, “I am quite convinced.”"
    )


def test_trim_to_complete_sentence_does_not_cut_at_prompt_abbreviation():
    text = "Mrs. Bennet was exceedingly"

    assert trim_to_complete_sentence(text, prompt="Mrs. Bennet") == text
