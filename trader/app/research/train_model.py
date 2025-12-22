import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from trader.app.research.build_dataset import build_dataframe


FEATURE_COLS = [
    "order_flow_imbalance",
    "volume_spike_ratio",
    "trade_intensity",
    "realized_volatility",
    "vwap_deviation",
    "z_score",
]


def train():
    df = build_dataframe().dropna()

    X = df[FEATURE_COLS]
    y = df["win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Validation accuracy: {score:.3f}")

    joblib.dump(model, "signal_gate_model.joblib")


if __name__ == "__main__":
    train()
