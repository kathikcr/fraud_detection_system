from fastapi.testclient import TestClient

from Src.api import app, create_app


def test_health_endpoint_reports_process_liveness():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["content-type"].startswith("application/json")


def test_health_is_the_only_route_in_this_api_checkpoint():
    application = create_app()
    paths = {route.path for route in application.routes}

    assert paths == {"/health"}


def test_health_endpoint_rejects_other_methods_and_unknown_paths():
    with TestClient(app) as client:
        method_response = client.post("/health")
        missing_response = client.get("/not-a-route")

    assert method_response.status_code == 405
    assert missing_response.status_code == 404
