from trader.app.common.db import get_db_session
from trader.app.common.models import SignalLog, PnL
from sqlalchemy import and_


def fetch_labeled_signals():
    """
    Join signal_logs to realized pnl.
    Each signal is matched to the next closed trade
    for the same symbol + strategy.
    """
    with get_db_session() as db:
        rows = (
            db.query(
                SignalLog.signal_id,
                SignalLog.symbol,
                SignalLog.strategy,
                SignalLog.features,
                SignalLog.z_score,
                SignalLog.entry_price,
                SignalLog.timestamp.label("signal_ts"),
                PnL.net_pnl,
                PnL.pnl_bps,
                PnL.exit_reason,
                PnL.timestamp.label("exit_ts"),
            )
            .join(
                PnL,
                and_(
                    PnL.symbol == SignalLog.symbol,
                    PnL.strategy_name == SignalLog.strategy,
                    PnL.timestamp >= SignalLog.timestamp,
                ),
            )
            .all()
        )

        return rows
