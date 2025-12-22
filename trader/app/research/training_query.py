from trader.app.common.supabase_client import get_client


def fetch_labeled_signals():
    """
    Join signal_logs to realized pnl.
    Each signal is matched to the next closed trade
    for the same symbol + strategy.
    """
    supabase = get_client()

    # 1. Fetch all signals
    signal_resp = (
        supabase
        .table("signal_logs")
        .select(
            "signal_id, symbol, strategy, features, z_score, entry_price, timestamp"
        )
        .execute()
    )

    signals = signal_resp.data or []
    if not signals:
        return []

    # 2. Fetch all pnl rows
    pnl_resp = (
        supabase
        .table("pnl")
        .select(
            "symbol, strategy_name, net_pnl, pnl_bps, exit_reason, timestamp"
        )
        .execute()
    )

    pnls = pnl_resp.data or []
    if not pnls:
        return []

    # 3. Index pnl by (symbol, strategy)
    pnl_by_key = {}
    for p in pnls:
        key = (p["symbol"], p["strategy_name"])
        pnl_by_key.setdefault(key, []).append(p)

    # Ensure pnl rows are time-sorted
    for key in pnl_by_key:
        pnl_by_key[key].sort(key=lambda x: x["timestamp"])

    # 4. Join in Python (signal → first pnl after signal)
    rows = []

    for s in signals:
        key = (s["symbol"], s["strategy"])
        if key not in pnl_by_key:
            continue

        signal_ts = s["timestamp"]

        for p in pnl_by_key[key]:
            if p["timestamp"] >= signal_ts:
                rows.append(
                    {
                        "signal_id": s["signal_id"],
                        "symbol": s["symbol"],
                        "strategy": s["strategy"],
                        "features": s["features"],
                        "z_score": s["z_score"],
                        "entry_price": s["entry_price"],
                        "signal_ts": signal_ts,
                        "net_pnl": p["net_pnl"],
                        "pnl_bps": p["pnl_bps"],
                        "exit_reason": p["exit_reason"],
                        "exit_ts": p["timestamp"],
                    }
                )
                break

    return rows
