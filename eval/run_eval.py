"""Compare exponential smoothing, XGBoost and the temporal GNN on the TEST simulation.
Run from repo root:  python -m eval.run_eval"""
import json, pandas as pd
from model.config import *
from model.data import load_csv
from model.baselines import ewm_predict, ewm_threshold, fit_xgb, xgb_predict, HAVE_XGB
from model.gnn import load_model, predict_df
from eval.metrics import pr_metrics, lead_times


def main():
    train, test = load_csv(TRAIN_CSV), load_csv(TEST_CSV)
    gnn, meta = load_model()
    W = meta["window"]

    xgb, xgb_thr = fit_xgb(train)
    models = {
        "Exp. smoothing": (ewm_predict(test), ewm_threshold(train)),
        ("XGBoost" if HAVE_XGB else "GradientBoosting"): (xgb_predict(xgb, test), xgb_thr),
        "Temporal GNN": (predict_df(gnn, test, meta), meta["threshold"]),
    }

    # Same rows for every model: hours with a full GNN window, and not already out of stock.
    base = test[(test["hour"] >= test["hour"].min() + W - 1) & (test["stockout_now"] == 0)]
    rows = []
    for name, (preds, thr) in models.items():
        d = base.merge(preds, on=["hour", "node_id", "item"], how="inner")
        m = pr_metrics(d["stockout_next_N"].values, d["prob"].values, thr)
        m.update(lead_times(test[test["hour"] >= test["hour"].min() + W - 1], preds, thr, N_HOURS))
        m["model"] = name; rows.append(m)

    res = pd.DataFrame(rows).set_index("model")[
        ["precision", "recall", "f1", "pr_auc", "detection_rate", "mean_lead_hrs", "events"]].round(3)
    print(f"\nLabel: stockout within next {N_HOURS}h | scored on {len(base)} rows, "
          f"{int(base['stockout_next_N'].sum())} positives\n")
    print(res.to_string())
    res.to_csv(ART_DIR / "results.csv"); res.to_json(ART_DIR / "results.json", indent=2)

    try:                                                   # PR curve picture for the video / README
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        plt.figure(figsize=(5, 4))
        for name, (preds, _) in models.items():
            d = base.merge(preds, on=["hour", "node_id", "item"])
            p, r, _ = precision_recall_curve(d["stockout_next_N"], d["prob"]); plt.plot(r, p, label=name)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("Stockout prediction (test)")
        plt.tight_layout(); plt.savefig(ART_DIR / "pr_curve.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()