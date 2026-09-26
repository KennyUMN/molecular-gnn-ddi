"""Single-thread CPU latency benchmark for the exported ONNX binaries.

Produces runs_kaggle/onnx_latency.json so the README/SPEC latency claim is
traceable to a real, reproducible number instead of an uncited constant.

    python bench_onnx_latency.py
"""
import json
import os
import time

import numpy as np
import onnxruntime as ort

MODELS = {
    "random": "runs_kaggle/random/molecular_gnn_ddi.onnx",
    "scaffold": "runs_kaggle/scaffold/molecular_gnn_ddi.onnx",
    "cold_start": "runs_kaggle/cold_start/molecular_gnn_ddi.onnx",
}
RUNS = 1000


def make_feed(rng, n_atoms, n_bonds):
    x = rng.standard_normal((n_atoms, 24)).astype(np.float32)
    ei = rng.integers(0, n_atoms, size=(2, n_bonds)).astype(np.int64)
    b = np.zeros(n_atoms, dtype=np.int64)
    return {"x1": x, "edge_index1": ei, "batch1": b,
            "x2": x.copy(), "edge_index2": ei.copy(), "batch2": b.copy()}


def main():
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    rng = np.random.default_rng(0)
    out = {}
    for split, path in MODELS.items():
        if not os.path.exists(path):
            continue
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        feed = make_feed(rng, 21, 40)  # aspirin-sized pair
        for _ in range(20):
            sess.run(None, feed)
        t = time.perf_counter()
        for _ in range(RUNS):
            sess.run(None, feed)
        ms = (time.perf_counter() - t) * 1000 / RUNS
        out[split] = {"model": path, "runs": RUNS, "single_thread_ms_per_pair": round(ms, 4),
                      "pairs_per_second": round(1000.0 / ms, 1)}
    with open("runs_kaggle/onnx_latency.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
