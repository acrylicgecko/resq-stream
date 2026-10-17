"""Export the trained GNN to ONNX and check it.   python -m model.export_onnx"""
import time, numpy as np, torch
from .config import *
from .gnn import load_model


def main():
    model, meta = load_model()
    Nn, F_ = len(meta["nodes"]), meta["in_f"]
    x = torch.randn(1, meta["window"], Nn, F_)
    path = ART_DIR / "resq_gnn.onnx"
    kw = dict(input_names=["x"], output_names=["logits"],
              dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, x, str(path), dynamo=False, **kw)   # newer torch: legacy exporter is the safest
    except TypeError:
        torch.onnx.export(model, x, str(path), **kw)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    xb = torch.randn(8, meta["window"], Nn, F_)
    ref = model(xb).detach().numpy()
    out = sess.run(None, {"x": xb.numpy()})[0]
    print("max |torch - onnx| =", float(np.abs(ref - out).max()))

    x1 = x.numpy(); sess.run(None, {"x": x1})                      # warm-up
    t0 = time.perf_counter()
    for _ in range(200):
        sess.run(None, {"x": x1})
    ms = (time.perf_counter() - t0) / 200 * 1000
    print(f"ONNX Runtime latency: {ms:.2f} ms per window (whole network, batch=1)")
    print("input  x      :", [meta["window"], Nn, F_], "= [W, nodes, features]")
    print("output logits :", [Nn, len(meta["items"])], "-> apply sigmoid; alert if >=", round(meta["threshold"], 3))
    print("saved", path)


if __name__ == "__main__":
    main()