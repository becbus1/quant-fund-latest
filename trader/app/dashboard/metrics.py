import pandas as pd
from sqlalchemy.orm import Session

from trader.app.common.models import SignalLog, PnL


def load_dashboard_dataframe(db: Session) -> pd.DataFrame:
    rows = (
        db.query(
            SignalLog.signal_id,
            SignalLog.symbol,
            SignalLog.strategy,
            SignalLog.z_score,
            SignalLog.features,
            PnL.pnl_bps,
            PnL.net_pnl,
        )
        .join(
            PnL,
            (PnL.symbol == SignalLog.symbol)
            & (PnL.strategy_name == SignalLog.strategy)
            & (PnL.timestamp >= SignalLog.timestamp),
        )
        .all()
    )

    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        rec = {
            "signal_id": r.signal_id,
            "symbol": r.symbol,
            "strategy": r.strategy,
            "z_score": r.z_score,
            "pnl_bps": r.pnl_bps,
            "net_pnl": r.net_pnl,
        }
        rec.update(r.features)
        records.append(rec)

    return pd.DataFrame(records)


def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "no_data"}

    df = df.copy()

    df["p_win_bucket"] = pd.cut(
        df["z_score"],
        bins=[-10, -2, -1, 0, 1, 2, 10],
        labels=["<-2", "-2:-1", "-1:0", "0:1", "1:2", ">2"],
    )

    return {
        "summary": {
            "trades": int(len(df)),
            "avg_pnl_bps": float(df["pnl_bps"].mean()),
            "win_rate": float((df["pnl_bps"] > 0).mean()),
        },
        "pnl_by_bucket": (
            df.groupby("p_win_bucket", observed=True)["pnl_bps"]
            .mean()
            .dropna()
            .to_dict()
        ),
        "trades_by_bucket": (
            df["p_win_bucket"].value_counts()
            .sort_index()
            .to_dict()
        ),
        "feature_correlations": (
            df.corr(numeric_only=True)["pnl_bps"]
            .drop("pnl_bps")
            .sort_values(ascending=False)
            .to_dict()
        ),
    }
