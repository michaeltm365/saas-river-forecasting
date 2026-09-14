"""Verify canonical tables directly against committed prediction snapshots."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from hja.paths import REPO
from hja.copula import load_hobo_daily
from hja.paper_results import sensor_predictions, raw_copula_table
from hja.evaluation import metrics, per_class


def main():
    root=REPO/'results/paper'
    saved=pd.read_csv(root/'classification_seeds.csv')
    truth=load_hobo_daily().rename(columns={'NHDPlusID':'site_id','Date':'target_date','wet':'sensor'})
    orders=pd.read_csv(root/'predictions/stream_order.csv').rename(columns={'NHDPlusID':'site_id'})
    for seed in (42,43,44):
        rg=sensor_predictions(seed).rename(columns={'date':'target_date','sensor_wet':'sensor'})
        ls=pd.read_csv(root/f'predictions/lstm_seed{seed}.csv',parse_dates=['target_date','issue_date'])
        ls=ls.merge(truth,on=['site_id','target_date'],validate='one_to_one').merge(orders,on='site_id',how='left',validate='many_to_one')
        common=rg[['site_id','target_date','issue_date']].merge(ls[['site_id','target_date','issue_date']],validate='one_to_one')
        assert len(common)==908
        for model,df in [('RGCN',rg),('LSTM',ls)]:
            np.testing.assert_array_equal(df.true_wetdry,df.sensor)
            for scope,g in [('full_sensor',df),('matched',df.merge(common)),('headwater',df[df.StreamOrde<=2]),('tailwater',df[df.StreamOrde>=3])]:
                y,p=g.sensor,g.pred_wetdry_prob
                m=metrics(y,p,p>=.5);m['DryPrecision']=precision_score(y,p>=.5,pos_label=0,zero_division=0)
                ref=saved[(saved.model==model)&(saved.seed==seed)&(saved.scope==scope)].iloc[0]
                for k,v in m.items():np.testing.assert_allclose(v,ref[k],atol=1e-12,rtol=1e-12)
    for model in ('lr','xgb'):
        report=json.loads((root/f'baselines/{model}_splits.json').read_text())
        for split,r in report.items():
            df=pd.read_csv(root/f'baselines/{model}_{split}_predictions.csv')
            check_baseline(df,r)
    r=json.loads((root/'baselines/lstm_hobo_temporal.json').read_text())
    df=pd.read_csv(root/'baselines/lstm_hobo_temporal_predictions.csv',parse_dates=['issue_date','target_date'])
    assert len(df)==742 and (df.issue_date>=pd.Timestamp('2020-09-15')).all() and (df.target_date>df.issue_date).all()
    check_baseline(df,r)
    raw=raw_copula_table().set_index('site').sort_index()
    ref=pd.read_csv(root/'copula/raw_seed42_copula.csv').set_index('site').sort_index()
    for col in ['n_observed','observed_dry','expected_dry','simulation_mean','lo','hi','rho']:
        np.testing.assert_allclose(raw[col],ref[col],atol=1e-10,rtol=1e-10)
    assert raw.included.sum()==18 and len(raw)==22
    sweep=pd.read_csv(root/'sensitivity.csv')
    assert len(sweep)==13 and sweep.Configuration.nunique()==13
    assert set(sweep.DischargeN_day1)=={1038} and set(sweep.DischargeN_day3)=={1054}
    for row in sweep.itertuples(index=False):
        folder=root/'sensitivity_predictions'/row.Configuration
        df=pd.read_csv(folder/'classification.csv')
        y,p=df.sensor_wet,df.pred_wetdry_prob
        got=metrics(y,p,p>=.5)
        expected=sweep[sweep.Configuration==row.Configuration].iloc[0]
        for k,v in got.items():np.testing.assert_allclose(v,expected[k],atol=1e-12)
        np.testing.assert_allclose(precision_score(y,p>=.5,pos_label=0),expected.DryPrecision,atol=1e-12)
        for day in (1,3):
            df=pd.read_csv(folder/f'discharge_day{day}.csv')
            t,p=df.true_discharge,df.pred_discharge
            nse=1-np.sum((t-p)**2)/np.sum((t-t.mean())**2)
            kge=1-np.sqrt((np.corrcoef(t,p)[0,1]-1)**2+(p.std(ddof=0)/t.std(ddof=0)-1)**2+(p.mean()/t.mean()-1)**2)
            np.testing.assert_allclose([nse,kge],[expected[f'NSE_day{day}'],expected[f'KGE_day{day}']],atol=1e-6)
    print('Verified neural classification, all HOBO model results, all 13 sensitivity configurations, and raw seed-42 copula.')


def check_baseline(df,r):
    m=metrics(df.true_wet,df.prob_wet,df.prob_wet>=.5)
    for k,v in m.items():np.testing.assert_allclose(v,r['metrics'][k],atol=1e-12)
    pc=per_class(df.true_wet,df.prob_wet>=.5)
    for cls in pc:
        for k,v in pc[cls].items():np.testing.assert_allclose(v,r['per_class'][cls][k],atol=1e-12)

if __name__=='__main__':main()
