"""Readable GPT model code adapted from chapters 2-7 of LLMs-from-scratch.

This package holds the reusable model, training, and generation code
(independent of any particular demo). The book's from-scratch ``GPTModel`` is
also the network that OpenAI's GPT-2 open weights load into — they ship as a
PyTorch state dict matching this model (see ``scripts/fetch_gpt2.py``), so the
chapters differ only in config preset and task, not architecture.
"""

