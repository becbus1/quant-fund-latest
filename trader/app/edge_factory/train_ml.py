from trader.app.edge_factory.data.load_training_data import load_training_data
from trader.app.edge_factory.features.flatten import flatten_features
from trader.app.edge_factory.models.logistic import train_logistic
from trader.app.edge_factory.models.xgboost_model import train_xgboost


def run_training():
    df = load_training_data()
    df = flatten_features(df)

    log_model, log_auc = train_logistic(df)
    print(f"Logistic AUC: {log_auc:.3f}")

    xgb_model, xgb_auc = train_xgboost(df)
    print(f"XGBoost AUC: {xgb_auc:.3f}")

    return {
        "logistic_auc": log_auc,
        "xgboost_auc": xgb_auc,
        "model": xgb_model if xgb_auc > log_auc else log_model,
    }


if __name__ == "__main__":
    run_training()
