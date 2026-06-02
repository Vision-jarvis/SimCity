"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data


class TestGraphEndpoints:
    def test_get_nodes(self):
        response = client.get("/graph/nodes?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_edges(self):
        response = client.get("/graph/edges?limit=5")
        assert response.status_code == 200

    def test_get_stats(self):
        response = client.get("/graph/stats")
        assert response.status_code == 200

    def test_get_communities(self):
        response = client.get("/graph/communities")
        assert response.status_code == 200

    def test_get_influence(self):
        response = client.get("/graph/influence?limit=5")
        assert response.status_code == 200


class TestTrendEndpoints:
    def test_current_trends(self):
        response = client.get("/trends/current")
        assert response.status_code == 200
        data = response.json()
        assert "trends" in data

    def test_forecast(self):
        response = client.post("/trends/forecast", json={
            "topic": "AI Safety",
            "horizon_hours": 12,
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 12

    def test_narratives(self):
        response = client.get("/trends/narratives")
        assert response.status_code == 200


class TestSearchEndpoints:
    def test_semantic_search(self):
        response = client.post("/search/semantic", json={
            "query": "artificial intelligence",
            "limit": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 5

    def test_nlp_analyze(self):
        response = client.get("/search/nlp/analyze?text=BREAKING%20news%20about%20AI")
        assert response.status_code == 200
        data = response.json()
        assert "analyses" in data


class TestSimulationEndpoints:
    def test_presets(self):
        response = client.get("/simulate/presets")
        assert response.status_code == 200
        data = response.json()
        assert "presets" in data
        assert len(data["presets"]) >= 3
