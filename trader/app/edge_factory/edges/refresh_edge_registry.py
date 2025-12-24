"""
Nightly Edge Registry Refresh Job

Responsibilities:
- Refresh hypothesis cube
- Promote statistically valid edges
- Pause decayed edges
- Retire dead edges

This is the brain of the Edge Factory.
"""

import logging
from trader.app.common.supabase_client import supabase

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_refresh():
    logger.info("Starting edge registry refresh")

    # 1️⃣ Refresh materialized view
    logger.info("Refreshing edge_cube materialized view")
    supabase.rpc("execute_sql", {
        "sql": "REFRESH MATERIALIZED VIEW public.edge_cube;"
    }).execute()

    # 2️⃣ Promote / refresh edges
    logger.info("Promoting qualifying edges")
    supabase.rpc("execute_sql", {
        "sql": """
        INSERT INTO public.edge_registry (
          symbol,
          strategy,
          side,
          z_score_bucket,
          volatility_bucket,
          status,
          n,
          wins,
          win_rate,
          avg_pnl_bps,
          lb95,
          last_refreshed_at
        )
        SELECT
          symbol,
          strategy,
          side,
          z_score_bucket,
          volatility_bucket,
          'active',
          n,
          wins,
          win_rate,
          avg_pnl_bps,
          lb95,
          now()
        FROM public.edge_cube_scored
        WHERE
          n >= 30
          AND lb95 >= 0.52
          AND avg_pnl_bps > 0
        ON CONFLICT (symbol, strategy, side, z_score_bucket, volatility_bucket)
        DO UPDATE SET
          n = EXCLUDED.n,
          wins = EXCLUDED.wins,
          win_rate = EXCLUDED.win_rate,
          avg_pnl_bps = EXCLUDED.avg_pnl_bps,
          lb95 = EXCLUDED.lb95,
          status = 'active',
          last_refreshed_at = now();
        """
    }).execute()

    # 3️⃣ Pause stale edges
    logger.info("Pausing stale edges")
    supabase.rpc("execute_sql", {
        "sql": """
        UPDATE public.edge_registry
        SET status = 'paused'
        WHERE
          status = 'active'
          AND last_refreshed_at < now() - interval '3 days';
        """
    }).execute()

    # 4️⃣ Retire dead edges
    logger.info("Retiring dead edges")
    supabase.rpc("execute_sql", {
        "sql": """
        UPDATE public.edge_registry
        SET status = 'retired'
        WHERE
          status IN ('active', 'paused')
          AND last_refreshed_at < now() - interval '14 days';
        """
    }).execute()

    logger.info("Edge registry refresh complete")


if __name__ == "__main__":
    run_refresh()
