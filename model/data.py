"""Loading, feature engineering and tensor building.
Everything here uses only PAST information at time t. The label (stockout_next_N) is never a feature,
and stockout_now is used only as a mask (we predict the *onset* of a stockout)."""
import numpy as np, pandas as pd
from .config import *


def load_csv(path):
    return pd.read_csv(path).sort_values(["node_id", "item", "hour"]).reset_index(drop=True)


def split_hours(T, frac=VAL_FRAC, gap=N_HOURS):
    """Time-ordered split: train hours < cut, val hours >= cut+gap (gap avoids label overlap)."""
    cut = int(T * (1 - frac))
    return cut, cut + gap


# ---------- tabular features (baselines) ----------
def tabular_features(df):
    df = df.sort_values(["node_id", "item", "hour"]).copy()
    g = df.groupby(["node_id", "item"])
    df["log_stock"] = np.log1p(df["stock"])
    df["hours_left"] = (df["stock"] / (df["consumption_rate"] + 1e-6)).clip(0, 72)
    df["d_stock_1"] = g["stock"].diff(1).fillna(0)
    df["d_stock_6"] = g["stock"].diff(6).fillna(0)
    df["rate_ma6"] = g["consumption_rate"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["rate_ewm"] = g["consumption_rate"].transform(lambda s: s.ewm(alpha=0.3).mean())
    df["hours_left_ewm"] = (df["stock"] / (df["rate_ewm"] + 1e-6)).clip(0, 72)
    df["has_inbound"] = (df["inbound_eta_hrs"] >= 0).astype(int)
    df["eta"] = df["inbound_eta_hrs"].where(df["inbound_eta_hrs"] >= 0, 72)
    df["cover_gap"] = df["hours_left"] - df["eta"]
    df["is_warehouse"] = (df["node_type"] == "warehouse").astype(int)
    for e in EVENTS:
        df["ev_" + e] = (df["event_flag"] == e).astype(int)
    df["hod_sin"] = np.sin(2 * np.pi * (df["hour"] % 24) / 24)
    df["hod_cos"] = np.cos(2 * np.pi * (df["hour"] % 24) / 24)
    items = sorted(df["item"].unique())
    df["item_code"] = df["item"].map({it: i for i, it in enumerate(items)})
    return df

TAB_FEATURES = ["log_stock", "consumption_rate", "hours_left", "d_stock_1", "d_stock_6", "rate_ma6",
                "hours_left_ewm", "has_inbound", "eta", "cover_gap", "is_warehouse", "hod_sin", "hod_cos",
                "item_code"] + ["ev_" + e for e in EVENTS]


# ---------- graph tensors (GNN) ----------
PER_ITEM = 6   # log_stock, log_rate, log_hours_left, eta, has_inbound, cover_gap

def build_tensors(df, nodes, items, with_labels=True):
    """Returns X [T,Nn,F] (+ Y, M [T,Nn,I] if with_labels). Hour index is relative to df.hour.min()."""
    h0 = int(df["hour"].min())
    T = int(df["hour"].max()) - h0 + 1
    Nn, I = len(nodes), len(items)
    n_of = {n: i for i, n in enumerate(nodes)}; i_of = {it: i for i, it in enumerate(items)}
    t = (df["hour"].values - h0).astype(int)
    n = df["node_id"].map(n_of).values; k = df["item"].map(i_of).values

    stock = df["stock"].values; rate = df["consumption_rate"].values; eta = df["inbound_eta_hrs"].values
    feat = np.zeros((T, Nn, I, PER_ITEM), dtype=np.float32)
    feat[t, n, k, 0] = np.log1p(stock) / 10.0
    feat[t, n, k, 1] = np.log1p(rate) / 5.0
    hl = np.clip(stock / (rate + 1e-6), 0, 72)
    eta_e = np.where(eta >= 0, np.clip(eta, 0, 72), 72)
    feat[t, n, k, 2] = np.log1p(hl) / np.log(73.0)          # log scale: low hours-left matters most
    feat[t, n, k, 3] = eta_e / 72.0
    feat[t, n, k, 4] = (eta >= 0).astype(np.float32)
    feat[t, n, k, 5] = np.clip(hl - eta_e, -72, 72) / 72.0   # negative = stock runs out before the shipment lands

    ev = np.zeros((T, Nn, len(EVENTS)), dtype=np.float32)
    for j, e in enumerate(EVENTS):
        m = (df["event_flag"].values == e)
        np.maximum.at(ev[:, :, j], (t[m], n[m]), 1.0)

    is_wh = np.zeros(Nn, dtype=np.float32)
    nt = df.drop_duplicates("node_id").set_index("node_id")["node_type"]
    for nid, i in n_of.items():
        is_wh[i] = float(nt[nid] == "warehouse")
    hod = (h0 + np.arange(T)) % 24
    tim = np.stack([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)], -1).astype(np.float32)

    X = np.concatenate([feat.reshape(T, Nn, I * PER_ITEM), ev,
                        np.broadcast_to(is_wh[None, :, None], (T, Nn, 1)),
                        np.broadcast_to(tim[:, None, :], (T, Nn, 2))], axis=-1).astype(np.float32)
    if not with_labels:
        return X
    Y = np.zeros((T, Nn, I), dtype=np.float32); M = np.zeros((T, Nn, I), dtype=np.float32)
    Y[t, n, k] = df["stockout_next_N"].values
    M[t, n, k] = 1.0 - df["stockout_now"].values     # only score rows that are NOT already out of stock
    return X, Y, M


def make_windows(X, W=WINDOW):
    """[T,Nn,F] -> [T-W+1, W, Nn, F]; window i ends at hour index i+W-1."""
    idx = np.arange(W)[None, :] + np.arange(X.shape[0] - W + 1)[:, None]
    return X[idx]


def build_adj(edges_df, nodes):
    """Symmetric adjacency + self loops, row-normalised (plain-PyTorch graph conv, no PyG needed)."""
    n_of = {n: i for i, n in enumerate(nodes)}
    A = np.eye(len(nodes), dtype=np.float32)
    for s, d in zip(edges_df["src"], edges_df["dst"]):
        A[n_of[s], n_of[d]] = 1.0; A[n_of[d], n_of[s]] = 1.0
    return A / A.sum(1, keepdims=True)


def graph_spec(df):
    return sorted(df["node_id"].unique()), sorted(df["item"].unique())