import os
from pathlib import Path

# Where Person 1's CSVs live. On the VM: export RESQ_DATA=/home/ibm-user/shared
DATA_DIR = Path(os.environ.get("RESQ_DATA", "data"))
ART_DIR = Path(os.environ.get("RESQ_ART", "model/artifacts"))
ART_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "telemetry_train.csv"
TEST_CSV = DATA_DIR / "telemetry_test.csv"
EDGES_CSV = DATA_DIR / "edges.csv"

N_HOURS = 6        # label horizon: stockout within next N hours (must match simulator N)
WINDOW = 12        # GNN looks at the last 12 hours
VAL_FRAC = 0.2     # last 20% of training hours used for validation
SEED = 42
EVENTS = ["outbreak_surge", "route_closure", "supplier_delay"]