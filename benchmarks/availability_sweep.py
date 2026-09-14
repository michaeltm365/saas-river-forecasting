"""Launch and score the q65 availability-aware, one-factor RGCN sweep."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess

import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/availability_sweep"
CONFIG=ROOT/"rgcn/availability_sweep"
PYTHON=str(ROOT/".venv/bin/python")


def stamp():return datetime.now(timezone.utc).isoformat()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CONFIG.mkdir(parents=True,exist_ok=True)
    specs=[]
    for path in sorted((ROOT/"rgcn/flagship/ablations_q65").glob("config_*.yml")):
        tag=path.stem.removeprefix("config_")
        cfg=yaml.safe_load(path.read_text())
        cfg["features"]["status_availability"]=True
        cfg["training"]["seed"]=42
        cfg["paths"]["checkpoint"]=f"data/retrain/availability_sweep/{tag}.pt"
        cfg["paths"]["predictions_dir"]=f"data/retrain/availability_sweep/predictions_{tag}"
        cfg["paths"]["eval_report"]=f"results/availability_sweep/{tag}.md"
        cfg["paths"]["hjflp_report"]=f"results/availability_sweep/{tag}_hjflp.md"
        dest=CONFIG/f"config_{tag}.yml"
        dest.write_text(yaml.safe_dump(cfg,sort_keys=False))
        specs.append(dict(tag=tag,config=str(dest.relative_to(ROOT))))
    assert len(specs)==15
    (OUT/"specs.json").write_text(json.dumps(specs,indent=2))
    print(f"Prepared {len(specs)} single-seed availability ablations",flush=True)


def worker(shard,gpu):
    specs=json.loads((OUT/"specs.json").read_text())[shard::2]
    state=dict(pid=os.getpid(),gpu=gpu,status="running",started=stamp(),completed=[])
    path=OUT/f"status_shard{shard}.json"
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",MPLCONFIGDIR="/tmp/mpl_availability")
    def save():path.write_text(json.dumps(state,indent=2))
    try:
        # Refresh Figure 3(d) without retraining; only the first shard does this.
        if shard==0:
            state["current"]="lstm_importance";save()
            with (OUT/"importance.log").open("x") as log:
                subprocess.run([PYTHON,"benchmarks/availability_products.py","--importance"],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for spec in specs:
            tag=spec["tag"];cfg=yaml.safe_load((ROOT/spec["config"]).read_text())
            if (ROOT/cfg["paths"]["checkpoint"]).exists():raise FileExistsError(f"Existing checkpoint {tag}; refusing overwrite")
            jobenv=env|{"RGCN_CONFIG":spec["config"]}
            for stage in ("train","export"):
                state.update(current=f"{tag}_{stage}",updated=stamp());save()
                cmd=[PYTHON,"-m","rgcn.pipeline.train"] if stage=="train" else [PYTHON,"-m","rgcn.pipeline.export_predictions","--eval-stride","1","--day3-range","2020-09-11:2020-12-31"]
                with (OUT/f"{tag}_{stage}.log").open("x") as log:
                    p=subprocess.Popen(cmd,cwd=ROOT,env=jobenv,stdout=log,stderr=subprocess.STDOUT)
                    state["child_pid"]=p.pid;save();code=p.wait()
                if code:raise RuntimeError(f"{tag}_{stage} failed: {code}")
                state["completed"].append(f"{tag}_{stage}");save()
        state.update(status="complete",current=None,finished=stamp())
    except Exception as exc:
        state.update(status="failed",error=str(exc),finished=stamp());raise
    finally:save()


def evaluate():
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import precision_score
    from correction_evaluate import sensors
    from hja.evaluation import metrics
    from availability_products import mdtable
    truth=sensors()
    specs=[dict(tag="default_availability",config="rgcn/correction_sep11/config_q65_availability_s42.yml"),
           dict(tag="no_availability",config="rgcn/flagship/config_q65.yml")]+json.loads((OUT/"specs.json").read_text())
    rows=[]
    for spec in specs:
        cfg=yaml.safe_load((ROOT/spec["config"]).read_text())
        folder=ROOT/(cfg["paths"]["predictions_dir"]+"_stride1")
        df=pd.read_csv(folder/"train_val_predictions_day3.csv",parse_dates=["date"]).rename(columns={"date":"target_date"})
        df=df[df.target_date>"2020-09-10"].merge(truth,on=["site_id","target_date"],validate="one_to_one")
        assert len(df)==908
        y,p=df.sensor.to_numpy(),df.pred_wetdry_prob.to_numpy()
        row=dict(Configuration=spec["tag"],**metrics(y,p,p>=.5))
        row["DryPrecision"]=precision_score(y,p>=.5,pos_label=0,zero_division=0)
        ck=torch.load(ROOT/cfg["paths"]["checkpoint"],map_location="cpu",weights_only=False)
        row["BestEpoch"]=ck["epoch"]+1
        for h in (1,3):
            dis=pd.read_csv(folder/f"train_val_predictions_day{h}.csv")
            dis=dis[dis.date>"2020-09-10"].dropna(subset=["true_discharge","pred_discharge"])
            t,p=dis.true_discharge.to_numpy(),dis.pred_discharge.to_numpy()
            row[f"NSE_day{h}"]=1-np.sum((t-p)**2)/np.sum((t-t.mean())**2)
            row[f"KGE_day{h}"]=1-np.sqrt((np.corrcoef(t,p)[0,1]-1)**2+(p.std()/t.std()-1)**2+(p.mean()/t.mean()-1)**2)
            row[f"DischargeN_day{h}"]=len(t)
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(OUT/"metrics.csv",index=False)
    (OUT/"SUMMARY.md").write_text("# Availability-aware RGCN sweep\n\nSeed 42, one factor per run. "
        "Classification uses the identical 908 sensor-verified daily t+3 targets; threshold 0.5. "
        "Discharge uses the same daily export date convention for all variants. "
        "Single-seed sensitivity does not establish equivalence or justify selecting a new optimum. "
        "Single-task rows include untrained other-head diagnostics.\n\n"+mdtable(table)+"\n")


def supervise(gpus):
    state=dict(pid=os.getpid(),status="running",started=stamp(),gpus=gpus,workers=[])
    children=[]
    for shard,gpu in enumerate(gpus):
        with (OUT/f"shard{shard}.log").open("x") as log:
            p=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),"--worker",str(shard),"--gpu",str(gpu)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        children.append(p);state["workers"].append(p.pid)
    (OUT/"status.json").write_text(json.dumps(state,indent=2))
    codes=[p.wait() for p in children]
    state["exit_codes"]=codes
    if any(codes):state["status"]="failed"
    else:
        try:evaluate();state["status"]="complete"
        except Exception as exc:state.update(status="failed",error=str(exc))
    state["finished"]=stamp();(OUT/"status.json").write_text(json.dumps(state,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--prepare",action="store_true");ap.add_argument("--launch",action="store_true")
    ap.add_argument("--worker",type=int);ap.add_argument("--gpu",type=int);ap.add_argument("--gpus",default="2,3")
    args=ap.parse_args()
    if args.prepare:prepare();return
    if args.worker is not None:worker(args.worker,args.gpu);return
    gpus=[int(x) for x in args.gpus.split(",")];assert len(gpus)==2
    if not args.launch:supervise(gpus);return
    if (OUT/"launch.json").exists():raise FileExistsError("Already launched")
    for gpu in gpus:
        subprocess.run([PYTHON,"-c","import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),check=True)
    with (OUT/"queue.log").open("x") as log:
        p=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),"--gpus",args.gpus],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    files=list(CONFIG.glob("*.yml"))+list((ROOT/"rgcn/pipeline").glob("*.py"))+[Path(__file__).resolve(),ROOT/"benchmarks/availability_products.py"]
    state=dict(pid=p.pid,gpus=gpus,launched=stamp(),new_training_runs=15,seed=42,sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (OUT/"launch.json").write_text(json.dumps(state,indent=2));print(f"Launched PID {p.pid}; GPUs {gpus}; 15 runs",flush=True)


if __name__=="__main__":main()
