import pandas as pd

def flatten_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand JSON features into numeric columns.
    """
    feature_df = pd.json_normalize(df["features"])
    feature_df.columns = [f"f_{c}" for c in feature_df.columns]

    base_cols = df[["z_score", "pnl_bps", "label"]].reset_index(drop=True)

    return pd.concat([base_cols, feature_df], axis=1)
