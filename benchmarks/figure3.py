"""Figure 3 using current canonical importance values and the Aug 31 layout.

Uses the shared category scheme introduced in commit 50e33bb. Historical
notebook model fitting and importance values are not used.
"""
from pathlib import Path
import hashlib
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from hja.data import build_hobo_frame
from hja.models import lr, xgb
from hja.importance import FEATURE_CATEGORIES, RENAME

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/paper/figure3'
INPUTS=OUT/'inputs'
# Colors and the two-by-two layout match the assembled Aug 31 figure.
COLORS={'Lagged Target':'#e876a4','Degrees':'#e69f00','Drivers':'#2c7bdc',
        'Static':'#008000','Other Obs':'#4b3aab','Other':'#777777'}
LABELS={
 'wetdry_status':'Recent wet/dry status','Out-degree':'Downstream connections',
 'In-degree':'Upstream connections','Slope':'Reach slope (%)',
 'slp_mean_pct':'Mean slope (%)','slp_median_pct':'Median slope (%)',
 'aspect_ne_pct':'NE aspect','aspect_nw_pct':'NW aspect',
 'aspect_se_pct':'SE aspect','aspect_sw_pct':'SW aspect',
 'elev_min_cm':'Min. elevation','elev_mean_cm':'Mean elevation',
 'elev_median_cm':'Median elevation','elev_max_cm':'Max. elevation',
 'curv_mean':'Mean curvature','curv_median':'Median curvature','LengthKM':'Reach length (km)',
 'sph':'Specific humidity','rhmin':'Min. rel. humidity','rhmax':'Max. rel. humidity',
 'tmin':'Min. temperature','tmax':'Max. temperature','ws':'Wind speed',
 'srad':'Solar radiation','prcp':'Precipitation','vp':'Vapor pressure',
 'etgrass':'Ref. ET (grass)','etalfalfa':'Ref. ET (alfalfa)',
 'MaxDepth_cm':'Maximum depth','MaxDepth_Threshold':'Detectable depth (0/1)',
 'MaxDepth_Censor':'Depth censor flag','status_available':'Status available (0/1)',
 'ArbolateSu':'Upstream channel length','AreaSqKm':'Catchment area','TotDASqKm':'Drainage area'}


def refresh_lstm_all():
    import torch
    from lstm_all_sites import sequences
    from hja.data import scale_train_only,BINARY_COLS
    from hja.models.lstm import LSTMModel
    from sklearn.metrics import f1_score
    torch.set_num_threads(4)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck=torch.load(ROOT/"data/retrain/paper/lstm_availability_s42.pt",map_location=device,weights_only=False)
    feats=ck["features"]; hp=ck["hyperparameters"]
    df=pd.read_parquet(ROOT/"data/retrain/paper/lstm_calendar_frame.parquet")
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
    saved=pd.read_csv(ROOT/"results/paper/predictions/lstm_seed42.csv")
    expected=saved[saved.label_is_hobo==1].pred_wetdry_prob.to_numpy()
    np.testing.assert_allclose(baseline,expected,atol=2e-5,rtol=2e-5)
    score=f1_score(y,baseline>=.5,pos_label=1,zero_division=0);rng=np.random.default_rng(42);rows=[]
    for j,name in enumerate(feats):
        losses=[]
        for repeat in range(5):
            xp=x.copy();xp[:,:,j]=x[rng.permutation(len(x)),:,j]
            losses.append(score-f1_score(y,predict(xp)>=.5,pos_label=1,zero_division=0))
        rows.append(dict(feature=name,f1_drop=np.mean(losses),sd=np.std(losses,ddof=1)))
    pd.DataFrame(rows).to_csv(INPUTS/"lstm_all_importance.csv",index=False)
    print(f"All-sites checkpoint verified: N={len(y)}, baseline wet F1={score:.9f}",flush=True)


def prepare():
    frame=build_hobo_frame()
    panels=[]
    for name,module in [('lr',lr),('xgb',xgb)]:
        r=module.run(frame,split='temporal',seed=42)
        reference=pd.read_csv(INPUTS/f'{name}_temporal_predictions.csv',dtype={'site':str})
        np.testing.assert_array_equal(r['sites'].astype(str),reference.site.astype(str))
        np.testing.assert_array_equal(r['y_true'],reference.true_wet)
        # XGBoost exports float32 probabilities with finite CSV precision.
        np.testing.assert_allclose(r['prob'],reference.prob_wet,rtol=1e-7,atol=1e-8)
        np.testing.assert_array_equal(r['prob'] >= .5, reference.prob_wet >= .5)
        df=r['importance'][['Feature','Value']].rename(columns={'Feature':'feature','Value':'raw_importance'})
        # Absolute coefficients for LR; gain for XGBoost.
        df['plot_importance']=df.raw_importance.abs() if name=='lr' else df.raw_importance
        df.to_csv(INPUTS/f'{name}_importance.csv',index=False)
        panels.append((name,df))
    hb=pd.read_csv(INPUTS/'lstm_hobo_importance.csv')
    hb=hb[['Feature','Value']].rename(columns={'Feature':'feature','Value':'raw_importance'})
    al=pd.read_csv(INPUTS/'lstm_all_importance.csv').rename(columns={'f1_drop':'raw_importance'})
    for name,df in [('lstm_hobo',hb),('lstm_all',al)]:
        # Negative decreases mean a permutation improved the score. Keep signed
        # values in the CSV and suppress them in the positive-importance plot.
        df['plot_importance']=df.raw_importance.clip(lower=0)
        panels.append((name,df))
    return panels


def draw(ax,df,title):
    top=df.sort_values(['plot_importance','feature'],ascending=[False,True]).head(10).copy()
    top['scaled_importance']=top.plot_importance/max(top.plot_importance.max(),1e-12)
    top=top.iloc[::-1]
    categories=[FEATURE_CATEGORIES.get(RENAME.get(f,f),'Other') for f in top.feature]
    ax.barh(range(len(top)),top.scaled_importance,color=[COLORS[c] for c in categories],height=.72)
    ax.set_yticks(range(len(top)),[LABELS.get(f,f) for f in top.feature],fontsize=9)
    ax.set_xlim(0,1.02);ax.set_xticks([0,.5,1],["0.0","0.5","1.0"])
    ax.tick_params(axis='both',length=0,labelsize=9)
    ax.xaxis.grid(True,color='#e5e5e5',lw=.7);ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title(title,loc='left',fontsize=11,fontweight='bold',pad=10)
    return top.iloc[::-1]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    panels=prepare()
    titles=['(a) Logistic Regression\n| standardized coefficient |',
            '(b) XGBoost\ngain-based importance',
            '(c) LSTM (HOBO only)\npermutation Δ F1',
            '(d) LSTM (all sites)\npermutation Δ F1']
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(13,8.2))
    displayed=[]
    for ax,(name,df),title in zip(axes.ravel(),panels,titles):
        top=draw(ax,df,title);top.insert(0,'panel',name);displayed.append(top)
        single,sax=plt.subplots(figsize=(7.2,4.8));draw(sax,df,title)
        single.supxlabel('Feature importance (0–1 max-scaled)',fontsize=10)
        single.tight_layout(rect=(0,.03,1,1))
        for ext in ['pdf','png','svg']:single.savefig(OUT/f'figure3_{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(single)
    legend_labels={'Lagged Target':'Recent water presence','Degrees':'Network degree',
                   'Drivers':'Meteorology','Static':'Static watershed','Other Obs':'Depth observations'}
    fig.legend(handles=[Patch(facecolor=COLORS[c],label=label) for c,label in legend_labels.items()],
               loc='upper center',bbox_to_anchor=(.5,1.015),ncol=5,frameon=False,fontsize=10)
    fig.supxlabel('Feature importance (0–1 max-scaled within each panel)',fontsize=11,y=.01)
    fig.tight_layout(rect=(0,.04,1,.96),h_pad=2.6,w_pad=3.3)
    for ext in ['pdf','png','svg']:fig.savefig(OUT/f'figure3.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    pd.concat(displayed).to_csv(OUT/'displayed_values.csv',index=False)
    info={'source_commit':'0dd101a','plotting_origin':'50e33bb individual notebooks; current src/hja/importance.py; composite layout reconstructed from Aug 31 Figure 3',
          'panels':{'a':'LR temporal seed 42, causal filling, coefficients; predicted probabilities verified against canonical snapshot',
                    'b':'XGBoost temporal seed 42, causal filling, gain; predicted probabilities verified against canonical snapshot',
                    'c':'HOBO LSTM temporal seed 42, N=742, one permutation per feature, wet F1 decrease; canonical saved importance',
                    'd':'All-sites availability LSTM seed 42, N=956 sensor targets, five permutations per feature, wet F1 decrease; rescored canonical checkpoint'},
          'negative_permutation_values':'Signed values retained in source CSV; negative values zeroed for ranking and plotting',
          'input_sha256':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS.iterdir())}}
    (OUT/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
    print(pd.concat(displayed)[['panel','feature','raw_importance','scaled_importance']].to_string(index=False))
    for ext in ('pdf', 'png'):
        shutil.copyfile(OUT/f'figure3_lstm_all.{ext}', OUT.parent/f'figure3d_lstm_availability.{ext}')
    shutil.copyfile(INPUTS/'lstm_all_importance.csv', OUT.parent/'lstm_importance.csv')
    print('Wrote combined figure and all four individual panels as PDF, PNG, and SVG.')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--refresh-lstm-all',action='store_true',help='Recalculate wet-F1 importance from the canonical all-sites checkpoint')
    args=parser.parse_args()
    if args.refresh_lstm_all:refresh_lstm_all()
    main()
