"""
Historical trade data fetcher for edge discovery.
Fetches trade data from Bybit REST API.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional

import requests
import pandas as pd

from shared.schemas import TradeData, Side

logger = logging.getLogger(__name__)

BYBIT_REST_URL = "https://api.bybit.com/v5/market/recent-trade"
BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"


class TradeFetcher:
    """Fetch historical trade data from Bybit."""

    def __init__(self, rate_limit_delay: float = 0.2):
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def fetch_recent_trades(
        self,
        symbol: str,
        limit: int = 1000,
    ) -> List[TradeData]:
        """Fetch most recent trades for a symbol."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": min(limit, 1000),
        }

        try:
            response = self.session.get(BYBIT_REST_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                logger.error(f"API error: {data.get('retMsg')}")
                return []

            trades = []
            for item in data.get("result", {}).get("list", []):
                trade = TradeData(
                    timestamp=datetime.fromtimestamp(int(item["time"]) / 1000),
                    symbol=symbol,
                    price=float(item["price"]),
                    quantity=float(item["size"]),
                    side=Side.BUY if item["side"] == "Buy" else Side.SELL,
                    trade_id=item["execId"],
                )
                trades.append(trade)

            return sorted(trades, key=lambda t: t.timestamp)

        except Exception as e:
            logger.error(f"Failed to fetch trades for {symbol}: {e}")
            return []

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Fetch kline (candlestick) data.
        Interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 200),
        }

        if start_time:
            params["start"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["end"] = int(end_time.timestamp() * 1000)

        try:
            response = self.session.get(BYBIT_KLINE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                logger.error(f"API error: {data.get('retMsg')}")
                return pd.DataFrame()

            records = []
            for item in data.get("result", {}).get("list", []):
                records.append({
                    "timestamp": datetime.fromtimestamp(int(item[0]) / 1000),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "turnover": float(item[6]),
                })

            df = pd.DataFrame(records)
            if not df.empty:
                df = df.sort_values("timestamp").reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_historical_klines(
        self,
        symbol: str,
        interval: str = "1",
        days: int = 7,
    ) -> pd.DataFrame:
        """Fetch historical klines by paginating backwards."""
        all_data = []
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        current_end = end_time

        while current_end > start_time:
            df = self.fetch_klines(
                symbol=symbol,
                interval=interval,
                end_time=current_end,
                limit=200,
            )

            if df.empty:
                break

            all_data.append(df)
            current_end = df["timestamp"].min() - timedelta(minutes=1)
            time.sleep(self.rate_limit_delay)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"])
        result = result.sort_values("timestamp").reset_index(drop=True)
        result = result[result["timestamp"] >= start_time]

        logger.info(f"Fetched {len(result)} klines for {symbol}")
        return result

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "1",
        days: int = 7,
    ) -> dict:
        """Fetch historical data for multiple symbols."""
        data = {}
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}...")
            df = self.fetch_historical_klines(symbol, interval, days)
            if not df.empty:
                data[symbol] = df
            time.sleep(self.rate_limit_delay)

        return data
