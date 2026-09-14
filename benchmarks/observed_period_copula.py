"""Validate aggregated rolling t+3 dry counts on actual sensor-observed dates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from correction_evaluate import sensors, read_rgcn, rgcn_path
from availability_products import mdtable
from hja.copula import load_hobo_daily, fit_platt, rho_lag1, observed_period_counts

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/canonical_availability'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    truth,daily=sensors(),load_hobo_daily()
    rows=[]; inputs=[]; summaries=[]
    for seed in (42,43,44):
        val=read_rgcn(seed,'availability').merge(truth,on=['site_id','target_date'],validate='one_to_one')
        assert len(val)==908
        assert ((val.target_date-val.issue_date).dt.days==3).all()
        tr=pd.read_csv(rgcn_path(seed,'availability',False)/'train_val_predictions_day3.csv',parse_dates=['date'])
        tr=tr.rename(columns={'date':'target_date'})
        tr=tr[tr.target_date<='2020-09-10'].merge(truth,on=['site_id','target_date'],validate='one_to_one')
        cal=fit_platt(tr.pred_wetdry_prob,tr.sensor)
        for method,calibration,cap in [('platt_rho98',cal,.98),('raw_unclipped',None,None)]:
            method_rows=[]
            for sid,g in val.groupby('site_id'):
                if len(g)<20:continue
                g=g.sort_values('target_date')
                pw=g.pred_wetdry_prob.to_numpy()
                if calibration is not None:pw=calibration(pw)
                pdry=1-pw
                rho,n_pairs=rho_lag1(daily,sid,lambda d:d<=pd.Timestamp('2020-09-10'))
                used=min(rho,cap) if cap is not None else rho
                counts=observed_period_counts(pdry,g.target_date,used,sid,seed)
                lo,hi=np.percentile(counts,[2.5,97.5])
                actual=int((g.sensor==0).sum())
                row=dict(seed=seed,method=method,site=int(sid),n_observed=len(g),
                         first_date=str(g.target_date.min().date()),last_date=str(g.target_date.max().date()),
                         calendar_span=(g.target_date.max()-g.target_date.min()).days+1,
                         observed_dry=actual,expected_dry=float(pdry.sum()),simulation_mean=float(counts.mean()),
                         lo=float(lo),hi=float(hi),included=bool(lo<=actual<=hi),rho=rho,rho_used=used,rho_pairs=n_pairs)
                method_rows.append(row);rows.append(row)
                for date,issue,y,p in zip(g.target_date,g.issue_date,g.sensor,pdry):
                    inputs.append(dict(seed=seed,method=method,site=int(sid),target_date=str(date.date()),
                                       issue_date=str(issue.date()),observed_dry=int(y==0),p_dry=float(p)))
            summaries.append(dict(seed=seed,method=method,reaches=len(method_rows),
                                  n_observed=sum(r['n_observed'] for r in method_rows),
                                  included=sum(r['included'] for r in method_rows),
                                  mean_interval_width=float(np.mean([r['hi']-r['lo'] for r in method_rows])),
                                  mean_absolute_count_error=float(np.mean([abs(r['expected_dry']-r['observed_dry']) for r in method_rows]))))
    df=pd.DataFrame(rows);sm=pd.DataFrame(summaries)
    df.to_csv(OUT/'observed_copula_sites.csv',index=False)
    sm.to_csv(OUT/'observed_copula_summary.csv',index=False)
    pd.DataFrame(inputs).to_csv(OUT/'observed_copula_inputs.csv',index=False)
    (OUT/'OBSERVED_COPULA.md').write_text(
        '# Dry counts over sensor-observed validation dates\n\n'
        'Canonical aggregation evaluation: availability-RGCN, q65, rolling exact-calendar t+3 forecasts. '
        'Only genuine sensor-labeled dates are counted. Each probability remains attached to its forecast target date; '
        'the latent AR(1) evolves across calendar gaps, but unobserved dates never enter the count. '
        '10,000 simulations per reach. Observed count is the actual number of dry sensor dates, without annualization. '
        'Predicted expected count is the sum of date-specific dry probabilities.\n\n'
        'Two setups: Platt calibration fitted on each model\'s own 579 training predictions with rho capped at 0.98; '
        'and raw probabilities with no cap. Rho is estimated from consecutive-day training sensor pairs; '
        'constant training series retain the existing rho=0 fallback. Seed 42 is the primary illustration, not selected by coverage.\n\n'
        +mdtable(sm)+'\n\n'
        '## Interpretation\n\n'
        'Intervals summarize rolling forecasts retrospectively; later issue dates can use observations acquired during validation. '
        'They are not a whole-period forecast issued on September 10. This limitation ALSO applied to the former 365-day setup: '
        'it bootstrapped these same rolling validation probabilities rather than forecasting a year from a single issue date. '
        'These conditional simulation intervals do not include parameter or calibration uncertainty; the binary-series correlation '
        'is used approximately as the latent Gaussian AR coefficient. Inclusion across 22 reaches is a descriptive diagnostic.\n\n'
        '## Seed-42 reach counts\n\n'+mdtable(df[df.seed==42])+ '\n')
    primary=df[(df.seed==42)&(df.method=='platt_rho98')].sort_values(['observed_dry','site'])
    order=primary.site.tolist();fig,ax=plt.subplots(figsize=(10,8))
    for method,offset,color,label in [('raw_unclipped',-.16,'#999999','Raw, no rho cap'),('platt_rho98',.16,'#2166ac','Platt + rho cap')]:
        g=df[(df.seed==42)&(df.method==method)].set_index('site').loc[order]
        y=np.arange(len(g))+offset
        ax.hlines(y,g.lo,g.hi,color=color,linewidth=2,label=label)
        ax.scatter(g.expected_dry,y,color=color,s=15)
    ax.scatter(primary.observed_dry,np.arange(len(primary)),marker='x',color='black',label='Observed dry count')
    ax.set_yticks(np.arange(len(primary)),[f'{r.site} (n={r.n_observed})' for r in primary.itertuples()])
    ax.set_xlabel('Dry days among sensor-observed validation dates');ax.legend();fig.tight_layout()
    fig.savefig(OUT/'observed_copula_intervals.png',dpi=180);fig.savefig(OUT/'observed_copula_intervals.pdf');plt.close(fig)
    (OUT/'observed_copula_manifest.json').write_text(json.dumps(dict(n_sims=10000,seeds=[42,43,44],cutoff='2020-09-10',
        annualization=False,probability_resampling=False,calendar_gaps_preserved=True,rolling_forecasts=True),indent=2))
    print(sm.to_string(index=False),flush=True)


if __name__=='__main__':main()
