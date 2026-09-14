"""Canonical availability-model tables, copula comparisons, and figures."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score

from correction_evaluate import read_rgcn, rgcn_path, sensors
from hja.copula import load_hobo_daily, fit_platt, site_row
from hja.evaluation import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/canonical_availability"


def mdtable(df):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def tables():
    truth = sensors()
    order = pd.read_csv(ROOT / "data/huggingface/nhd_id_stream_order_permanence.csv")
    order = order[["NHDPlusID", "StreamOrde"]].drop_duplicates()
    rows = []
    for seed in (42,43,44):
        rg = read_rgcn(seed, "availability").merge(truth, on=["site_id", "target_date"], validate="one_to_one")
        ls = pd.read_csv(ROOT / f"results/correction_sep11/lstm_availability/preds_q65_s{seed}.csv", parse_dates=["target_date", "issue_date"])
        ls = ls.merge(truth, on=["site_id", "target_date"], validate="one_to_one")
        keys = ["site_id", "target_date", "issue_date"]
        common = rg[keys].merge(ls[keys], on=keys, validate="one_to_one")
        for model, df in [("RGCN", rg), ("LSTM", ls)]:
            df = df.merge(order, left_on="site_id", right_on="NHDPlusID", how="left", validate="many_to_one")
            for scope, g in [("full_sensor", df), ("matched", df.merge(common, on=keys)),
                             ("headwater", df[df.StreamOrde <= 2]), ("tailwater", df[df.StreamOrde >= 3])]:
                p, y = g.pred_wetdry_prob.to_numpy(), g.sensor.to_numpy()
                m = metrics(y,p,p>=.5)
                m["DryPrecision"] = precision_score(y,p>=.5,pos_label=0,zero_division=0)
                rows.append(dict(model=model,scope=scope,seed=seed,**m))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "classification_seeds.csv", index=False)
    report = []
    for (model,scope), g in df.groupby(["model","scope"]):
        row = dict(Model=model,Scope=scope,N=int(g.N.iloc[0]))
        for metric in ["Accuracy","ROC-AUC","WetF1","DryPrecision","DryRecall","DryF1"]:
            row[metric] = f"{g[metric].mean():.3f} ± {g[metric].std(ddof=1):.3f}"
        report.append(row)
    (OUT / "TABLES.md").write_text(
        "# Canonical availability models\n\nq65, exact calendar t+3; sensor labels only. Mean ± sample SD over seeds 42/43/44. "
        "SD describes training variability, not sampling uncertainty.\n\n" + mdtable(pd.DataFrame(report)) + "\n")
    fig, axes = plt.subplots(1,2,figsize=(9,4))
    for ax, metric in zip(axes,["Accuracy","DryRecall"]):
        for i,model in enumerate(["LSTM","RGCN"]):
            g=df[(df.model==model)&(df.scope=="matched")]
            ax.scatter(np.full(len(g),i),g[metric],alpha=.7)
            ax.errorbar(i,g[metric].mean(),yerr=g[metric].std(),fmt="ks",capsize=5)
        ax.set_xticks([0,1],["LSTM + availability","RGCN + availability"])
        ax.set_ylabel(metric);ax.set_ylim(.85,1.01)
    fig.tight_layout();fig.savefig(OUT / "classification_seeds.png",dpi=180);plt.close(fig)


def copulas():
    truth, daily = sensors(), load_hobo_daily()
    results, summaries = [], []
    for seed in (42,43,44):
        val = read_rgcn(seed,"availability").merge(truth,on=["site_id","target_date"],validate="one_to_one")
        tr=pd.read_csv(rgcn_path(seed,"availability",False)/"train_val_predictions_day3.csv",parse_dates=["date"])
        tr=tr.rename(columns={"date":"target_date"})
        tr=tr[tr.target_date<="2020-09-10"].merge(truth,on=["site_id","target_date"],validate="one_to_one")
        cal=fit_platt(tr.pred_wetdry_prob,tr.sensor)
        for method, calibration, cap in [("platt_rho98",cal,.98),("raw_unclipped",None,None)]:
            rows=[]
            for sid,g in val.groupby("site_id"):
                if len(g)<20:continue
                r=site_row(sid,g.pred_wetdry_prob,g.sensor,daily,
                           lambda d:d<=pd.Timestamp("2020-09-10"),calibration,cap,seed=seed)
                rows.append(dict(seed=seed,method=method,**r))
            results.extend(rows)
            summaries.append(dict(seed=seed,method=method,sensor_rows=len(val),calibration_rows=len(tr) if calibration else 0,
                                  reaches=len(rows),covered=sum(r["ok"] for r in rows),
                                  mean_width=np.mean([r["hi"]-r["lo"] for r in rows])))
    df=pd.DataFrame(results);df.to_csv(OUT/"copula_sites.csv",index=False)
    sm=pd.DataFrame(summaries);sm.to_csv(OUT/"copula_summary.csv",index=False)
    (OUT/"COPULA.md").write_text(
        "# Historical annualized availability-RGCN copula comparison\n\nCurrent observed-period evaluation: OBSERVED_COPULA.md. This annualized setup also aggregates rolling t+3 forecasts.\n\nBoth setups use identical exact sensor keys and 10,000 simulations/site. "
        "Current setup: logistic calibration fitted on the model's own training predictions and rho capped at 0.98. "
        "Old setup: raw probabilities, no rho cap. All three model seeds are shown; seed 42 remains the primary illustration.\n\n"
        "Counts are annualized late-season equivalents under stationary resampling, not observed annual counts. "
        "Interval inclusion is exploratory; the binary-series correlation is used as a latent Gaussian AR coefficient.\n\n"
        +mdtable(sm)+"\n")
    order=df[(df.seed==42)&(df.method=="platt_rho98")].sort_values("true").site.tolist()
    fig,ax=plt.subplots(figsize=(10,8))
    for method,offset,color in [("raw_unclipped",-.15,"#999999"),("platt_rho98",.15,"#2166ac")]:
        g=df[(df.seed==42)&(df.method==method)].set_index("site").loc[order]
        y=np.arange(len(g))+offset
        ax.hlines(y,g.lo,g.hi,color=color,linewidth=2,label=method)
        ax.scatter(g["mean"],y,color=color,s=12)
    ax.scatter(g["true"],np.arange(len(g)),color="black",marker="x",label="Observed fraction × 365")
    ax.set_yticks(np.arange(len(g)),[str(s) for s in order]);ax.set_xlabel("Annualized dry-day equivalents")
    ax.legend();fig.tight_layout();fig.savefig(OUT/"copula_intervals.png",dpi=180);fig.savefig(OUT/"copula_intervals.pdf");plt.close(fig)
    print(sm.to_string(index=False),flush=True)


def importance():
    import torch
    from correction_lstm import sequences
    from hja.data import scale_train_only,BINARY_COLS
    from hja.models.lstm import LSTMModel
    from hja.importance import importance_frame,plot_top10
    torch.set_num_threads(4)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck=torch.load(ROOT/"data/retrain/correction_sep11/lstm_availability_s42.pt",map_location=device,weights_only=False)
    feats=ck["features"]; hp=ck["hyperparameters"]
    df=pd.read_parquet(ROOT/"data/retrain/correction_sep11/lstm_calendar_frame.parquet")
    BINARY_COLS.add("status_available")
    _,scaled=scale_train_only(df.loc[df.target_date<="2020-09-10",feats],df[feats],feats)
    for c in feats:df[c]=scaled[c].fillna(0).values
    x,y,sites,dates,hobo=sequences(df,feats)
    take=(dates>pd.Timestamp("2020-09-10"))&(hobo==1)
    x,y=x[take],y[take]
    m=LSTMModel(len(feats),hp["hidden"],hp["layers"],hp["dropout"]).to(device)
    m.load_state_dict(ck["model_state_dict"]);m.eval()
    def predict(a):
        with torch.no_grad():return torch.sigmoid(m(torch.tensor(a,device=device))).cpu().numpy().ravel()
    baseline=predict(x)
    saved=pd.read_csv(ROOT/"results/correction_sep11/lstm_availability/preds_q65_s42.csv")
    expected=saved[saved.label_is_hobo==1].pred_wetdry_prob.to_numpy()
    np.testing.assert_allclose(baseline,expected,atol=2e-5,rtol=2e-5)
    score=accuracy_score(y,baseline>=.5);rng=np.random.default_rng(42);rows=[]
    for j,name in enumerate(feats):
        losses=[]
        for repeat in range(5):
            xp=x.copy();xp[:,:,j]=x[rng.permutation(len(x)),:,j]
            losses.append(score-accuracy_score(y,predict(xp)>=.5))
        rows.append(dict(feature=name,accuracy_drop=np.mean(losses),sd=np.std(losses,ddof=1)))
    result=pd.DataFrame(rows);result.to_csv(OUT/"lstm_importance.csv",index=False)
    # Use signed decreases for ranking; do not turn accuracy gains into positive importance.
    display=importance_frame(result.feature,result.accuracy_drop.clip(lower=0))
    ax=plot_top10(display,"LSTM + availability (sensor validation, seed 42)")
    ax.figure.tight_layout();ax.figure.savefig(OUT/"figure3d_lstm_availability.png",dpi=180)
    ax.figure.savefig(OUT/"figure3d_lstm_availability.pdf");plt.close(ax.figure)
    (OUT/"IMPORTANCE.md").write_text(
        "# LSTM feature importance\n\nSeed 42, 956 sensor targets. Five sequence-level permutations per feature, "
        "measured by decrease in accuracy. Saved checkpoint predictions reproduced before scoring. "
        "Signed values are retained in CSV; negative decreases are clipped to zero only for the max-scaled figure. "
        "Correlated inputs and overlapping sequences limit causal interpretation.\n")
    print("Importance complete; checkpoint predictions reproduced",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--importance",action="store_true");args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.importance:importance()
    else:tables();copulas()
