import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, average_precision_score


def best_threshold(y, p):
    """Threshold with the best F1 (choose it on VALIDATION data, never on test)."""
    best_t, best_f = 0.5, -1.0
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.999, 120))):
        _, _, f, _ = precision_recall_fscore_support(y, (p >= t).astype(int), average="binary", zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def pr_metrics(y, p, thr):
    pr, rc, f1, _ = precision_recall_fscore_support(y, (p >= thr).astype(int), average="binary", zero_division=0)
    return dict(precision=pr, recall=rc, f1=f1, pr_auc=average_precision_score(y, p) if y.sum() else float("nan"))


def lead_times(df, preds, thr, N=6):
    """df: test telemetry; preds: hour,node_id,item,prob.
    For every stockout onset (stockout_now goes 0->1) look at the N hours before it and find the earliest hour
    where the model already alerted (prob>=thr). lead = onset_hour - earliest_alert_hour.
    Returns detection_rate, mean_lead_hours (over detected events), n_events."""
    d = df.merge(preds, on=["hour", "node_id", "item"], how="left").sort_values(["node_id", "item", "hour"])
    leads, n_events = [], 0
    for _, g in d.groupby(["node_id", "item"]):
        so = g["stockout_now"].values; hr = g["hour"].values; pr = g["prob"].values
        for i in range(1, len(g)):
            if so[i] == 1 and so[i - 1] == 0:                    # onset of a stockout
                n_events += 1
                alerts = [hr[i] - hr[j] for j in range(max(0, i - N), i)
                          if not np.isnan(pr[j]) and pr[j] >= thr]
                if alerts:
                    leads.append(max(alerts))
    return dict(events=n_events,
                detection_rate=(len(leads) / n_events) if n_events else float("nan"),
                mean_lead_hrs=float(np.mean(leads)) if leads else 0.0)