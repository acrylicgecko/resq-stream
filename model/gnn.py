"""Temporal GNN in plain PyTorch (no PyG needed, so it installs on s390x and exports to ONNX cleanly).
Per hour: each node mixes its features with its neighbours' (graph step). A GRU then reads the last W hours.
Two more graph layers mix the GRU summaries across the network. Output: stockout logit per node and item."""
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from .config import *
from .data import build_tensors, make_windows


class TemporalGNN(nn.Module):
    def __init__(self, A, in_f, n_items, hidden=32):
        super().__init__()
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))
        self.Nn, self.H = int(self.A.shape[0]), hidden
        self.inp = nn.Linear(2 * in_f, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.g1 = nn.Linear(2 * hidden, hidden)
        self.g2 = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, n_items)
        self.skip = nn.Linear(in_f, n_items)                 # direct path from the latest readings
        self.drop = nn.Dropout(0.2)

    def forward(self, x):                                   # x: [B, W, Nn, F]
        nb = torch.matmul(self.A, x)                         # neighbour average at every hour
        h = torch.relu(self.inp(torch.cat([x, nb], -1)))     # [B, W, Nn, H]
        W = h.shape[1]
        h = h.permute(0, 2, 1, 3).reshape(-1, W, self.H)     # [B*Nn, W, H]
        _, last = self.gru(h)                                # [1, B*Nn, H]
        z = last[0].reshape(-1, self.Nn, self.H)             # [B, Nn, H]
        z = torch.relu(self.g1(torch.cat([z, torch.matmul(self.A, z)], -1)))
        z = torch.relu(self.g2(torch.cat([z, torch.matmul(self.A, z)], -1))) + z
        return self.out(self.drop(z)) + self.skip(x[:, -1])  # logits [B, Nn, n_items]


def load_model(art_dir=ART_DIR):
    meta = json.load(open(art_dir / "meta.json"))
    A = np.load(art_dir / "adj.npy")
    m = TemporalGNN(A, meta["in_f"], len(meta["items"]), meta["hidden"])
    m.load_state_dict(torch.load(art_dir / "gnn.pt", map_location="cpu"))
    return m.eval(), meta


@torch.no_grad()
def predict_df(model, df, meta, bs=128):
    """Probabilities for every hour that has a full window. Returns hour,node_id,item,prob."""
    nodes, items, W = meta["nodes"], meta["items"], meta["window"]
    X = build_tensors(df, nodes, items, with_labels=False)
    Xw = torch.from_numpy(make_windows(X, W))
    P = torch.cat([torch.sigmoid(model(Xw[i:i + bs])) for i in range(0, len(Xw), bs)]).numpy()  # [Tw,Nn,I]
    h0 = int(df["hour"].min()); Tw, Nn, I = P.shape
    hours = (h0 + W - 1 + np.arange(Tw))
    return pd.DataFrame({"hour": np.repeat(hours, Nn * I),
                         "node_id": np.tile(np.repeat(nodes, I), Tw),
                         "item": np.tile(np.array(items * Nn), Tw),
                         "prob": P.reshape(-1)})