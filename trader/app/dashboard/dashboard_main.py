from fastapi import FastAPI
from trader.app.dashboard.routes import router as dashboard_router

app = FastAPI(
    title="Trading Metrics Dashboard API",
    version="1.0.0",
)

# IMPORTANT: do NOT init DB at import time
app.include_router(dashboard_router)

@app.get("/health")
def health():
    return {"status": "ok"}
