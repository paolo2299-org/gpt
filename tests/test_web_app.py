from __future__ import annotations


def test_index_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Prompt to continue" in response.data
    assert b"Continuation" in response.data


def test_prompt_completion(client):
    response = client.post(
        "/",
        data={
            "prompt": "It was a truth",
            "max_new_tokens": "12",
            "temperature": "0",
            "top_k": "50",
            "seed": "123",
        },
    )

    assert response.status_code == 200
    assert b"It was a truth continued with 12 tokens" in response.data


def test_empty_prompt_shows_error(client):
    response = client.post("/", data={"prompt": ""})

    assert response.status_code == 200
    assert b"Enter a prompt to continue." in response.data

