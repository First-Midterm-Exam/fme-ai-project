from pydantic import SecretStr

from app.core.config import Settings

URL = "/api/v1/documentos/revision-formato"


def post_image(client, content, content_type="image/png", headers=None):
    return client.post(
        URL,
        files={"imagen": ("doc.png", content, content_type)},
        headers=headers or {},
    )


def test_returns_laravel_contract(client_factory, complete_reviewer,
                                  png_bytes):
    with client_factory(complete_reviewer) as client:
        response = post_image(client, png_bytes)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "cumple", "puntaje", "tipo_detectado", "resumen", "hallazgos",
    }
    assert body["cumple"] is True
    assert body["puntaje"] == 100


def test_rejects_non_image(client_factory, complete_reviewer):
    with client_factory(complete_reviewer) as client:
        response = post_image(client, b"texto", "text/plain")

    assert response.status_code == 415


def test_rejects_oversized_image(client_factory, complete_reviewer):
    with client_factory(complete_reviewer, max_upload_mb=0) as client:
        response = post_image(client, b"x")

    assert response.status_code == 413


def test_invalid_image_content_returns_422(client_factory,
                                           complete_reviewer):
    with client_factory(complete_reviewer) as client:
        response = post_image(client, b"no es png")

    assert response.status_code == 422


def test_requires_api_key_when_configured(client_factory, complete_reviewer,
                                          png_bytes):
    api_key = SecretStr("secreta")
    with client_factory(complete_reviewer, api_key=api_key) as client:
        denied = post_image(client, png_bytes)
        allowed = post_image(client, png_bytes, headers={
            "X-API-Key": "secreta",
        })

    assert denied.status_code == 401
    assert allowed.status_code == 200


def test_missing_model_returns_503_and_degraded_health(client_factory,
                                                       png_bytes):
    with client_factory() as client:
        response = post_image(client, png_bytes)
        health = client.get("/health").json()

    assert response.status_code == 503
    assert health["status"] == "degraded"
    assert health["model_loaded"] is False


def test_empty_api_key_in_env_disables_auth(monkeypatch):
    monkeypatch.setenv("API_KEY", "")

    assert Settings().api_key is None
