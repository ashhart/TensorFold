import httpx

from tensorfold_web import mock_engine


def test_model_lookup_returns_the_listed_entry(monkeypatch):
    monkeypatch.setattr(mock_engine, "MODEL", "org/test-model")
    monkeypatch.setattr(mock_engine.State, "plain", True)
    with mock_engine.serve(port=0, api_key="test-key") as server:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with httpx.Client(base_url=base, trust_env=False, headers={"Authorization": "Bearer test-key"}) as client:
                listing = client.get("/v1/models")
                assert listing.status_code == 200
                [model] = listing.json()["data"]
                assert model == {"id": "org/test-model", "object": "model", "owned_by": "library"}
                for route in ("/v1/models/org/test-model", "/models/org%2Ftest-model?unused=1"):
                    response = client.get(route)
                    assert response.status_code == 200
                    assert response.json() == model
                assert client.get("/compat/models").json() == listing.json()

                unknown = client.get("/v1/models/other%2Fmodel")
                assert unknown.status_code == 404
                assert unknown.json()["error"] == {
                    "message": "model 'other/model' is not served here",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
                assert client.get("/v1/models/org/test-model", headers={"Authorization": ""}).status_code == 401
        finally:
            server.shutdown()
