from fastapi import APIRouter
from trader.app.common.db import get_db_session
from trader.app.dashboard.metrics import load_dashboard_dataframe, compute_metrics

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("")
def dashboard():
    with get_db_session() as db:
        df = load_dashboard_dataframe(db)
        metrics = compute_metrics(df)
        return metrics
