"""
SimCity API — Main FastAPI application.
Aggregates all routers and middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from api.routers import graph, simulation, trends, search, websocket
from api.graphql.schema import schema

app = FastAPI(
    title="SimCity — AI Digital Twin of the Internet",
    description="Real-time simulation engine for internet behavior, virality, and influence cascades.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === REST Routers ===
app.include_router(graph.router)
app.include_router(simulation.router)
app.include_router(trends.router)
app.include_router(search.router)
app.include_router(websocket.router)

# === GraphQL ===
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")


# === Health ===
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "simcity-api",
        "version": "1.0.0",
    }


@app.get("/")
async def root():
    """API root."""
    return {
        "name": "SimCity API",
        "version": "1.0.0",
        "docs": "/docs",
        "graphql": "/graphql",
        "endpoints": {
            "health": "/health",
            "graph": "/graph/",
            "simulate": "/simulate/",
            "trends": "/trends/",
            "search": "/search/",
            "websocket": "/ws/stream",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
