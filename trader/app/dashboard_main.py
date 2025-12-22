"""
Dashboard-only FastAPI service.
Read-only API for metrics, PnL, and monitoring.
SAFE to expose publicly.
"""

from fastapi import FastAPI
from trader.app.common.db import init_db
from trader.app.dashboard.routes import router as dashboard_router

# Create dashboard-only app
app = FastAPI(
    title="Trading Metrics Dashboard API",
    description="Read-only analytics and performance metrics",
    version="1.0.0",
)

# Initialize DB (read-only usage)
init_db()

# Include dashboard routes
app.include_router(dashboard_router)


@app.get("/health")
async def health_check():
    return {
        "service": "dashboard",
        "status": "ok",
    }
