from fastapi.testclient import TestClient

from Src.api import app, create_app


def test_health_endpoint_reports_process_liveness():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["content-type"].startswith("application/json")


def test_api_route_surface_contains_only_completed_endpoints():
    application = create_app()
    paths = {route.path for route in application.routes}

    assert paths == {"/", "/dashboard/overview", "/health", "/predict", "/investigate"}


def test_local_dashboard_shell_is_directly_accessible_without_login():
    with TestClient(create_app()) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="dataset-select"' in response.text
    assert 'id="kpi-total"' in response.text
    assert 'aria-label="Dashboard navigation"' in response.text
    assert "no sign-in required" in response.text.lower()
    assert 'type="password"' not in response.text.lower()


def test_health_endpoint_rejects_other_methods_and_unknown_paths():
    with TestClient(app) as client:
        method_response = client.post("/health")
        missing_response = client.get("/not-a-route")

    assert method_response.status_code == 405
    assert missing_response.status_code == 404
