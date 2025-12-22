from fastapi import APIRouter
from trader.app.common.supabase_client import supabase

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/pnl")
def pnl_summary():
    """
    Aggregate PnL by date from Supabase.
    """
    resp = (
        supabase
        .table("pnl")
        .select("timestamp, net_pnl")
        .order("timestamp")
        .execute()
    )

    rows = resp.data or []

    summary = {}
    for r in rows:
        date = r["timestamp"][:10]  # YYYY-MM-DD
        summary.setdefault(date, {"net_pnl": 0.0, "trades": 0})
        summary[date]["net_pnl"] += float(r["net_pnl"])
        summary[date]["trades"] += 1

    return [
        {
            "date": d,
            "net_pnl": round(v["net_pnl"], 4),
            "trades": v["trades"],
        }
        for d, v in sorted(summary.items())
    ]


@router.get("/signals")
def signals(limit: int = 100):
    resp = (
        supabase
        .table("signal_logs")
        .select("symbol, strategy, z_score, entry_price, timestamp")
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )

    return resp.data or []


@router.get("/positions")
def positions():
    resp = (
        supabase
        .table("positions")
        .select(
            "symbol, side, quantity, entry_price, strategy_name, status"
        )
        .execute()
    )

    rows = resp.data or []

    return [
        {
            "symbol": p["symbol"],
            "side": p["side"],
            "qty": p["quantity"],
            "entry_price": p["entry_price"],
            "strategy": p["strategy_name"],
            "status": p["status"],
        }
        for p in rows
    ]
