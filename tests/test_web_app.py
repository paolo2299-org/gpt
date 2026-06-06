from __future__ import annotations


def test_landing_lists_available_demos(client):
    # Default client injects a runner for "jane" only; "gpt2" has no weights.
    response = client.get("/")

    assert response.status_code == 200
    assert b"LLM From Scratch Demos" in response.data
    assert b"JaneGPT" in response.data
    assert b"A homemade LLM trained from scratch" in response.data
    assert b"GPT-2 (124M)" not in response.data  # weights absent -> hidden


def test_landing_shows_demo_when_weights_present(make_client):
    client = make_client(
        runners={}, files=["model.dickens.pth", "gpt2-small.pth", "gpt2-medium.pth"]
    )

    response = client.get("/")

    assert response.status_code == 200
    assert b"JaneGPT" in response.data
    assert b"GPT-2 (124M)" in response.data
    assert b"GPT-2 (355M)" in response.data


def test_demo_page_loads(client):
    response = client.get("/jane")

    assert response.status_code == 200
    assert b"Start of phrase" in response.data
    assert b"JaneGPT" in response.data
    assert b"A homemade LLM trained solely on the works of Jane Austen" in response.data
    assert b"terabytes of data and millions of dollars" in response.data
    assert b"Add the start of a phrase..." in response.data
    assert b"All demos" in response.data  # back link to landing
    assert b"It was a truth" in response.data
    assert b"The morning was" in response.data
    assert b"She had never" in response.data
    assert b"In the drawing-room" in response.data
    assert b'name="max_new_tokens"' not in response.data
    assert b"Temperature" not in response.data
    assert b"Top K" not in response.data
    assert b"Seed" not in response.data
    assert b'class="spinner"' in response.data


def test_prompt_completion(client):
    response = client.post("/jane", data={"prompt": "It was a truth"})

    assert response.status_code == 200
    assert b"Completed phrase" in response.data
    assert b"It was a truth continued with 80 tokens" in response.data
    assert b"The morning was" not in response.data


def test_empty_prompt_shows_error(client):
    response = client.post("/jane", data={"prompt": ""})

    assert response.status_code == 200
    assert b"Enter a prompt to continue." in response.data


def test_unknown_demo_returns_404(client):
    assert client.get("/does-not-exist").status_code == 404


def test_unavailable_demo_returns_404(client):
    # gpt2 has neither an injected runner nor a weights file in the default client.
    assert client.get("/gpt2").status_code == 404
