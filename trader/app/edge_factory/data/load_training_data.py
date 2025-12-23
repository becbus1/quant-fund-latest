import pandas as pd
from trader.app.common.supabase_client import supabase

def load_training_data(min_rows: int = 500) -> pd.DataFrame:
    """
    Load ML training data from Supabase view.
    """
    response = (
        supabase
        .table("ml_training_data")
        .select("*")
        .execute()
    )

    if not response.data or len(response.data) < min_rows:
        raise ValueError(
            f"Not enough training data: {len(response.data)} rows"
        )

    df = pd.DataFrame(response.data)

    # Convert label to int (sklearn requirement)
    df["label"] = df["label"].astype(int)

    return df
