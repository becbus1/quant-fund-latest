import pandas as pd

df = pd.read_csv("ml_eval_results.csv")

def evaluate(threshold):
    subset = df[df["p_win"] >= threshold]
    if len(subset) == 0:
        return None

    return {
        "threshold": threshold,
        "trades": len(subset),
        "win_rate": (subset["pnl_bps"] > 0).mean(),
        "avg_pnl_bps": subset["pnl_bps"].mean(),
        "total_pnl_bps": subset["pnl_bps"].sum(),
    }

results = []
for t in [0.5, 0.55, 0.6, 0.65, 0.7]:
    r = evaluate(t)
    if r:
        results.append(r)

out = pd.DataFrame(results)
print(out)
