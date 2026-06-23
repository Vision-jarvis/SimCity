"""
Locust load test for the SimCity API (Phase 6 — production hardening).

Exercises the read-heavy graph/trends endpoints plus the heavier
simulation and forecast POST endpoints, weighted to approximate a realistic
dashboard traffic mix.

Run against a local API::

    pip install locust
    uvicorn api.main:app  # in another shell
    locust -f scripts/locustfile.py --host http://localhost:8000

Headless (CI smoke / benchmark)::

    locust -f scripts/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 1m --csv .locust_results/run

The Phase 3/6 scalability targets to validate are documented in
``internet twin.md`` (e.g. inference latency < 2s, then < 500ms).
"""

from locust import HttpUser, task, between


class DashboardUser(HttpUser):
    """Simulates a user browsing the live dashboard (read-heavy)."""

    wait_time = between(1, 4)

    @task(5)
    def health(self):
        self.client.get("/health", name="/health")

    @task(8)
    def graph_nodes(self):
        self.client.get("/graph/nodes?limit=100", name="/graph/nodes")

    @task(6)
    def graph_edges(self):
        self.client.get("/graph/edges?limit=100", name="/graph/edges")

    @task(4)
    def graph_stats(self):
        self.client.get("/graph/stats", name="/graph/stats")

    @task(4)
    def communities(self):
        self.client.get("/graph/communities", name="/graph/communities")

    @task(6)
    def trends(self):
        self.client.get("/trends/current", name="/trends/current")

    @task(3)
    def narratives(self):
        self.client.get("/trends/narratives", name="/trends/narratives")


class AnalystUser(HttpUser):
    """Simulates an analyst running forecasts, searches, and simulations."""

    wait_time = between(2, 6)

    @task(4)
    def semantic_search(self):
        self.client.post(
            "/search/semantic",
            json={"query": "misinformation cascade", "top_k": 10},
            name="/search/semantic",
        )

    @task(3)
    def forecast(self):
        self.client.post(
            "/trends/forecast",
            json={"topic": "ai-regulation", "horizon_hours": 24, "platforms": ["reddit", "hn"]},
            name="/trends/forecast",
        )

    @task(2)
    def run_simulation(self):
        # Small step count keeps the load-test request bounded.
        self.client.post(
            "/simulate/run",
            json={"scenario_id": "loadtest", "steps": 20, "N": 50000},
            name="/simulate/run",
        )
