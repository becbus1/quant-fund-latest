import pandas as pd
from trader.app.research.training_query import fetch_labeled_signals


def build_dataframe():
    rows = fetch_labeled_signals()

    records = []
    for r in rows:
        base = {
            "signal_id": r.signal_id,
            "symbol": r.symbol,
            "strategy": r.strategy,
            "z_score": r.z_score,
            "pnl_bps": r.pnl_bps,
            "win": 1 if r.pnl_bps > 0 else 0,
        }
        base.update(r.features)  # expand JSON features
        records.append(base)

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = build_dataframe()
    print(df.head())
    print(df.describe())
