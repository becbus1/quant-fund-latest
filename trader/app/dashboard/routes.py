from fastapi import APIRouter
from sqlalchemy import func

from trader.app.common.db import get_db_session
from trader.app.common.models import PnL, SignalLog, Position

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/pnl")
def pnl_summary():
    with get_db_session() as db:
        rows = (
            db.query(
                func.date(PnL.timestamp).label("date"),
                func.sum(PnL.net_pnl).label("net_pnl"),
                func.count().label("trades"),
            )
            .group_by(func.date(PnL.timestamp))
            .order_by(func.date(PnL.timestamp))
            .all()
        )

        return [
            {
                "date": r.date,
                "net_pnl": float(r.net_pnl),
                "trades": r.trades,
            }
            for r in rows
        ]


@router.get("/signals")
def signals(limit: int = 100):
    with get_db_session() as db:
        rows = (
            db.query(SignalLog)
            .order_by(SignalLog.timestamp.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "symbol": s.symbol,
                "strategy": s.strategy,
                "z_score": s.z_score,
                "entry_price": s.entry_price,
                "timestamp": s.timestamp,
            }
            for s in rows
        ]


@router.get("/positions")
def positions():
    with get_db_session() as db:
        rows = db.query(Position).all()

        return [
            {
                "symbol": p.symbol,
                "side": p.side.value,
                "qty": p.quantity,
                "entry_price": p.entry_price,
                "strategy": p.strategy_name,
                "status": p.status.value,
            }
            for p in rows
        ]
