import os
import requests
from typing import Dict, Any

from supabase import create_client  # ✅ ADDED

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("Supabase environment variables not set")

# ✅ EXPORT SUPABASE CLIENT (for dashboard + future use)
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

_HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}


def insert_row(table: str, payload: Dict[str, Any]) -> None:
    """
    Insert a single row into a Supabase table via REST API.
    """
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    response = requests.post(url, json=payload, headers=_HEADERS, timeout=10)

    if not response.ok:
        raise RuntimeError(
            f"Supabase insert failed [{response.status_code}]: {response.text}"
        )
