"""Current RGCN paper evaluation and raw observed-period copula illustration."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

from hja.paths import REPO
from hja.copula import load_hobo_daily, rho_lag1, observed_period_counts


def sensor_predictions(seed=42):
    folder=REPO/f"data/retrain/correction_sep11/predictions_rgcn_availability_s{seed}_stride1"
    df=pd.read_csv(folder/'train_val_predictions_day3.csv',parse_dates=['date'])
    df=df[df.date>'2020-09-10'].copy()
    truth=load_hobo_daily().rename(columns={'NHDPlusID':'site_id','Date':'date','wet':'sensor_wet'})
    df=df.merge(truth,on=['site_id','date'],validate='one_to_one')
    order=pd.read_csv(REPO/'data/huggingface/nhd_id_stream_order_permanence.csv')[['NHDPlusID','StreamOrde']].drop_duplicates()
    df=df.merge(order,left_on='site_id',right_on='NHDPlusID',how='left',validate='many_to_one')
    df['issue_date']=df.date-pd.Timedelta(days=3)
    assert len(df)==908
    np.testing.assert_array_equal(df.true_wetdry.round(),df.sensor_wet)
    return df


def classification(df):
    y,p=df.sensor_wet.to_numpy(),df.pred_wetdry_prob.to_numpy()
    precision,recall,f1,support=precision_recall_fscore_support(y,p>=.5,labels=[0,1],zero_division=0)
    return dict(N=len(y),Accuracy=accuracy_score(y,p>=.5),ROC_AUC=roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan,
                DryPrecision=precision[0],DryRecall=recall[0],DryF1=f1[0],
                WetPrecision=precision[1],WetRecall=recall[1],WetF1=f1[1])


def raw_copula_table(seed=42,n_sims=10000):
    """No probability calibration, no rho cap; count only actual sensor dates."""
    df=sensor_predictions(seed);daily=load_hobo_daily();rows=[]
    for site,g in df.groupby('site_id'):
        if len(g)<20:continue
        g=g.sort_values('date');p=1-g.pred_wetdry_prob.to_numpy()
        rho,pairs=rho_lag1(daily,site,lambda d:d<=pd.Timestamp('2020-09-10'))
        counts=observed_period_counts(p,g.date,rho,site,seed,n_sims)
        lo,hi=np.percentile(counts,[2.5,97.5]);actual=int((g.sensor_wet==0).sum())
        rows.append(dict(site=int(site),n_observed=len(g),observed_dry=actual,expected_dry=float(p.sum()),
                         simulation_mean=float(counts.mean()),lo=float(lo),hi=float(hi),rho=rho,rho_pairs=pairs,
                         included=bool(lo<=actual<=hi)))
    return pd.DataFrame(rows)


def plot_raw_intervals(table):
    df=table.sort_values(['observed_dry','site']);y=np.arange(len(df))
    fig,ax=plt.subplots(figsize=(10,8))
    ax.hlines(y,df.lo,df.hi,color='#2166ac',linewidth=2,label='95% simulation interval')
    ax.scatter(df.expected_dry,y,color='#2166ac',s=18,label='Expected dry count')
    ax.scatter(df.observed_dry,y,color='black',marker='x',label='Observed dry count')
    ax.set_yticks(y,[f'{r.site} (n={r.n_observed})' for r in df.itertuples()])
    ax.set_xlabel('Dry days among sensor-observed validation dates')
    ax.set_title('RGCN: raw probabilities, observed-period counts (seed 42)')
    ax.legend();fig.tight_layout();return fig


def plot_raw_map(table):
    """Use existing local geographic caches; never download during evaluation."""
    cache=REPO/'data/huggingface/map_cache'
    geom=cache/'hja_flowline_geoms_4326.json';hill=cache/'hja_tinted_hillshade.png'
    if not geom.exists() or not hill.exists():
        raise FileNotFoundError('Local map caches are required for the map; interval plots need no map cache.')
    geoms={int(k):v for k,v in json.loads(geom.read_text()).items()}
    points=np.concatenate([np.array(p) for paths in geoms.values() for p in paths])
    low,high=points.min(0),points.max(0);pad=(high-low)*.05
    bbox=[low[0]-pad[0],high[0]+pad[0],low[1]-pad[1],high[1]+pad[1]]
    fig,ax=plt.subplots(figsize=(12,9));ax.imshow(plt.imread(hill),extent=bbox,aspect='auto')
    for paths in geoms.values():
        for p in paths:
            a=np.array(p);ax.plot(a[:,0],a[:,1],color='#1f5fd0',lw=.7)
    mapped=0
    for r in table.itertuples():
        if int(r.site) not in geoms:continue
        pts=np.concatenate([np.array(p) for p in geoms[int(r.site)]]);x,y=pts.mean(0)
        ax.scatter(x,y,s=100,c='#198754' if r.included else '#c62828',edgecolors='white',zorder=5)
        mapped+=1
    if mapped!=len(table):raise ValueError(f'Only {mapped}/{len(table)} evaluated reaches have map geometry')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],marker='o',linestyle='',color='#198754',label='Observed count inside interval'),
                       Line2D([],[],marker='o',linestyle='',color='#c62828',label='Observed count outside interval')])
    ax.set_xlim(bbox[:2]);ax.set_ylim(bbox[2:]);ax.set_aspect(1/np.cos(np.deg2rad((bbox[2]+bbox[3])/2)))
    ax.set_xlabel('Longitude');ax.set_ylabel('Latitude')
    ax.set_title('Observed-period dry counts: raw RGCN probabilities (seed 42)')
    fig.tight_layout();return fig
