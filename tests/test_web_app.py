from __future__ import annotations


def test_index_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Start of phrase" in response.data
    assert b"JaneGPT" in response.data
    assert b"trained solely on the complete works of Jane Austen" in response.data
    assert b"Add the start of a phrase..." in response.data
    assert b"Start with a sentence fragment" not in response.data
    assert b"It was a truth" in response.data
    assert b"The morning was" in response.data
    assert b"She had never" in response.data
    assert b"In the drawing-room" in response.data
    assert b"name=\"max_new_tokens\"" not in response.data
    assert b"Temperature" not in response.data
    assert b"Top K" not in response.data
    assert b"Seed" not in response.data
    assert b"class=\"spinner\"" in response.data


def test_prompt_completion(client):
    response = client.post(
        "/",
        data={
            "prompt": "It was a truth",
        },
    )

    assert response.status_code == 200
    assert b"Completed phrase" in response.data
    assert b"It was a truth continued with 80 tokens" in response.data
    assert b"The morning was" not in response.data


def test_empty_prompt_shows_error(client):
    response = client.post("/", data={"prompt": ""})

    assert response.status_code == 200
    assert b"Enter a prompt to continue." in response.data
