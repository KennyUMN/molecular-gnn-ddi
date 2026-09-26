"""Full Kaggle pipeline: build the TDC dataset, then train all 3 splits on 2x T4.

Launch detached from the Kaggle kernel:
    nohup python -u kaggle_run.py > logs/pipeline.log 2>&1 &

GPU layout: two concurrent single-GPU runs (nn.DataParallel cannot scatter
graph edge lists): GPU 0 chains random -> cold_start, GPU 1 runs scaffold.
Same wall clock as a sequential run on both GPUs, twice the early coverage.

`python kaggle_run.py chain <gpu> <split,split>` runs one chain (spawning mode).
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
DATA = "data/tdc_drugbank_ddi.csv"
EPOCHS = "30"


def ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
        return True
    except ImportError:
        r = subprocess.run([PY, "-m", "pip", "install", "-q", pkg], capture_output=True, text=True)
        print(f"pip install {pkg} rc={r.returncode} {r.stderr[-200:]}", flush=True)
        return r.returncode == 0


def train(split, gpu):
    cmd = [PY, "-u", "kaggle_train.py", "--split", split, "--data", DATA,
           "--epochs", EPOCHS, "--workers", "2", "--out", f"runs/{split}"]
    print(f"[gpu{gpu}] train {split}: {' '.join(cmd)}", flush=True)
    with open(f"logs/train_{split}.log", "w") as lf:
        return subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))


def chain(gpu, splits):
    for s in splits:
        t0 = time.time()
        r = train(s, gpu)
        print(f"[gpu{gpu}] {s} rc={r.returncode} in {time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0:
            tail = open(f"logs/train_{s}.log").read()[-3000:]
            print(f"[gpu{gpu}] {s} FAILED:\n{tail}", flush=True)
            return 1
    return 0


def main():
    os.chdir(ROOT)
    os.makedirs("logs", exist_ok=True)

    for pkg, imp in (("rdkit", "rdkit"), ("PyTDC", "tdc"), ("onnx", "onnx"),
                     ("onnxruntime", "onnxruntime"), ("onnxscript", "onnxscript")):
        ensure(pkg, imp)

    if not os.path.exists(DATA):
        print("building dataset (TDC DrugBank + 1:1 sampled negatives)...", flush=True)
        with open("logs/build.log", "w") as lf:
            r = subprocess.run([PY, "-u", "data/build_tdc_dataset.py", "--out", DATA],
                               stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(open("logs/build.log").read()[-3000:], flush=True)
            sys.exit("dataset build failed")
        print("dataset built", flush=True)

    procs = [
        subprocess.Popen([PY, "-u", __file__, "chain", "0", "random,cold_start"],
                         stdout=open("logs/gpu0_chain.log", "w"), stderr=subprocess.STDOUT,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"), start_new_session=True),
        subprocess.Popen([PY, "-u", __file__, "chain", "1", "scaffold"],
                         stdout=open("logs/gpu1_chain.log", "w"), stderr=subprocess.STDOUT,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"), start_new_session=True),
    ]
    rcs = []
    for p in procs:
        rcs.append(p.wait())
    if all(rc == 0 for rc in rcs):
        print("ALL DONE", flush=True)
    else:
        print(f"PIPELINE FAILED chain exit codes: {rcs}", flush=True)
    sys.exit(0 if all(rc == 0 for rc in rcs) else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chain":
        os.chdir(ROOT)
        os.makedirs("logs", exist_ok=True)
        sys.exit(chain(sys.argv[2], sys.argv[3].split(",")) or 0)
    else:
        main()
