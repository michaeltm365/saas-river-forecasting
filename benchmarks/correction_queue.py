"""Detached, fail-fast queue for the nine-run September 11 correction campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/correction_sep11"
PYTHON = str(ROOT / ".venv/bin/python")


def now():
    return datetime.now(timezone.utc).isoformat()


def save(state):
    temp = OUT / "status.tmp"
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(OUT / "status.json")


def work(gpu):
    os.chdir(ROOT)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    state = dict(pid=os.getpid(), started=now(), gpu=gpu, status="running", completed=[], current=None)
    save(state)
    jobs = []
    # Seed 42 of each new variant first; remaining seeds always follow.
    for seed in (42, 43, 44):
        for variant in ("control", "availability"):
            jobs.append((f"lstm_{variant}_s{seed}",
                         [PYTHON, "benchmarks/correction_lstm.py", "--seed", str(seed), "--variant", variant], {}))
        cfg = f"rgcn/correction_sep11/config_q65_availability_s{seed}.yml"
        cfg_env = {"RGCN_CONFIG": cfg}
        for stage in ("train", "export_predictions", "eval_report"):
            jobs.append((f"rgcn_availability_s{seed}_{stage}",
                         [PYTHON, "-m", f"rgcn.pipeline.{stage}"], cfg_env))
        jobs.append((f"rgcn_availability_s{seed}_daily_export",
                     [PYTHON, "-m", "rgcn.pipeline.export_predictions", "--eval-stride", "1",
                      "--day3-range", "2020-09-11:2020-12-31"], cfg_env))
    jobs.append(("matched_evaluation", [PYTHON, "benchmarks/correction_evaluate.py"], {}))
    try:
        for name, command, extra_env in jobs:
            state["current"] = name
            state["updated"] = now()
            save(state)
            print(f"{now()} START {name}", flush=True)
            with (OUT / f"{name}.log").open("x") as log:
                proc = subprocess.Popen(command, cwd=ROOT, env=env | extra_env,
                                        stdout=log, stderr=subprocess.STDOUT)
                state["child_pid"] = proc.pid
                save(state)
                code = proc.wait()
            if code:
                raise RuntimeError(f"{name} exited {code}; see {name}.log")
            state["completed"].append(name)
            print(f"{now()} DONE {name}", flush=True)
        state.update(status="complete", current=None, finished=now())
    except Exception as exc:
        state.update(status="failed", error=str(exc), finished=now())
        raise
    finally:
        save(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.launch:
        work(args.gpu)
        return
    if (OUT / "launch.json").exists():
        raise FileExistsError("Campaign already launched; inspect status.json rather than launching twice")
    # Check actual CUDA access before reporting a launch.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    subprocess.run([PYTHON, "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"],
                   env=env, check=True)
    sources = [Path(__file__), ROOT / "benchmarks/correction_lstm.py",
               ROOT / "benchmarks/lstm_flagship_splits.py", ROOT / "benchmarks/correction_evaluate.py"]
    sources += list((ROOT / "rgcn/pipeline").glob("*.py"))
    sources += list((ROOT / "rgcn/correction_sep11").glob("*.yml"))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with (OUT / "queue.log").open("x") as log:
        proc = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), "--gpu", str(args.gpu)],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    manifest = dict(pid=proc.pid, gpu=args.gpu, launched=now(), training_runs=9,
                    seeds=[42,43,44], source_sha256=hashes)
    (OUT / "launch.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k != "source_sha256"}), flush=True)


if __name__ == "__main__":
    main()
