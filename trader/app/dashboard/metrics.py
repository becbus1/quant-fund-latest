import pandas as pd

from trader.app.common.supabase_client import supabase


def load_dashboard_dataframe() -> pd.DataFrame:
    """
    Load joined SignalLog + PnL data from Supabase
    and return as a pandas DataFrame.
    """

    # Fetch signals
    signals_resp = (
        supabase
        .table("signal_logs")
        .select(
            "signal_id, symbol, strategy, z_score, features, timestamp"
        )
        .execute()
    )

    # Fetch PnL
    pnl_resp = (
        supabase
        .table("pnl")
        .select(
            "symbol, strategy_name, pnl_bps, net_pnl, timestamp"
        )
        .execute()
    )

    signals = signals_resp.data or []
    pnls = pnl_resp.data or []

    if not signals or not pnls:
        return pd.DataFrame()

    pnl_df = pd.DataFrame(pnls)
    signal_df = pd.DataFrame(signals)

    # Normalize timestamps
    pnl_df["timestamp"] = pd.to_datetime(pnl_df["timestamp"])
    signal_df["timestamp"] = pd.to_datetime(signal_df["timestamp"])

    # Join logic (equivalent to SQLAlchemy join)
    merged = signal_df.merge(
        pnl_df,
        left_on=["symbol", "strategy"],
        right_on=["symbol", "strategy_name"],
        suffixes=("", "_pnl"),
    )

    # Only keep PnL that happened AFTER the signal
    merged = merged[
        merged["timestamp_pnl"] >= merged["timestamp"]
    ]

    if merged.empty:
        return pd.DataFrame()

    records = []
    for _, r in merged.iterrows():
        rec = {
            "signal_id": r["signal_id"],
            "symbol": r["symbol"],
            "strategy": r["strategy"],
            "z_score": r["z_score"],
            "pnl_bps": r["pnl_bps"],
            "net_pnl": r["net_pnl"],
        }
        if isinstance(r["features"], dict):
            rec.update(r["features"])
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
            df["p_win_bucket"]
            .value_counts()
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
