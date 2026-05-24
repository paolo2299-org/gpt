from __future__ import annotations


def test_index_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Start of phrase" in response.data
    assert b"End of phrase" in response.data
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
    assert b"It was a truth continued with 80 tokens" in response.data


def test_empty_prompt_shows_error(client):
    response = client.post("/", data={"prompt": ""})

    assert response.status_code == 200
    assert b"Enter a prompt to continue." in response.data
