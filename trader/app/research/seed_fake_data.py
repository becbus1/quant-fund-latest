from datetime import datetime, timedelta
import random

from trader.app.common.db import get_db_session, init_db
from trader.app.common.models import SignalLog, PnL
from shared.schemas import Side


def seed():
    init_db()

    with get_db_session() as db:
        for i in range(200):
            ts = datetime.utcnow() - timedelta(minutes=i)

            features = {
                "order_flow_imbalance": random.uniform(-1, 1),
                "volume_spike_ratio": random.uniform(0.5, 3.0),
                "trade_intensity": random.uniform(1, 50),
                "realized_volatility": random.uniform(0.0005, 0.01),
                "vwap_deviation": random.uniform(-0.01, 0.01),
            }

            z = features["order_flow_imbalance"] * 3

            signal = SignalLog(
                symbol="ETHUSDT",
                strategy="smoke_test_time_exit",
                features=features,
                z_score=z,
                entry_price=3000,
                timestamp=ts,
            )

            pnl = PnL(
                timestamp=ts + timedelta(seconds=60),
                symbol="ETHUSDT",
                strategy_name="smoke_test_time_exit",
                entry_price=3000,
                exit_price=3000 * (1 + random.uniform(-0.003, 0.003)),
                quantity=0.01,
                side=Side.BUY,
                gross_pnl=random.uniform(-3, 3),
                fees=0.1,
                net_pnl=random.uniform(-3, 3),
                pnl_bps=random.uniform(-30, 30),
                hold_time_sec=60,
                exit_reason="test",
            )

            db.add(signal)
            db.add(pnl)


if __name__ == "__main__":
    seed()
