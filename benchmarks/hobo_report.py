"""Regenerate canonical HOBO metrics, predictions, and LSTM importance."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hja.paths import REPO
from hja.data import build_hobo_frame
from hja.models import lr, xgb, lstm_hobo
from hja.importance import importance_frame, plot_top10
from hja.models.tabular import write_report


def main():
    torch.set_num_threads(2)
    out=REPO/'results/paper/baselines';out.mkdir(parents=True,exist_ok=True)
    frame=build_hobo_frame()
    for name,module in [('lr',lr),('xgb',xgb)]:
        results=module.run_all(frame, seed=42)
        write_report(name, results, "Exact three-calendar-day observed targets; causal depth filling; seed 42; original split rules.")
        (out/f'{name}_splits.json').write_text(json.dumps({s:{'metrics':r['metrics'],'per_class':r['per_class']} for s,r in results.items()},indent=2)+'\n')
        for split,r in results.items():
            pd.DataFrame(dict(site=r['sites'],date=r['dates'],target_date=pd.to_datetime(r['dates']) + pd.Timedelta(days=3),true_wet=r['y_true'],prob_wet=r['prob'])).to_csv(out/f'{name}_{split}_predictions.csv',index=False)
    r=lstm_hobo.train_eval(device=torch.device('cpu'))
    (out/'lstm_hobo_temporal.json').write_text(json.dumps({'metrics':r['metrics'],'per_class':r['per_class'],'seed':42,'device':'cpu','torch_threads':2,'split_date':r['split_date'],'inner_cutoff':r['inner_cutoff']},indent=2)+'\n')
    pd.DataFrame(dict(site=r['sites_test'],target_date=r['target_dates_test'],issue_date=r['issue_dates_test'],true_wet=r['y_true'],prob_wet=r['prob'])).to_csv(out/'lstm_hobo_temporal_predictions.csv',index=False)
    feats,drops=lstm_hobo.permutation_importance(r)
    imp=importance_frame(feats,drops)
    imp.to_csv(out/'lstm_hobo_importance.csv',index=False)
    ax=plot_top10(imp,'HOBO-only LSTM: temporal evaluation')
    ax.figure.tight_layout()
    for ext in ['png','pdf']:ax.figure.savefig(REPO/f'results/paper/figure3c_lstm_hobo.{ext}',dpi=180)
    plt.close(ax.figure)
    figure_inputs=REPO/'results/paper/feature_importance/inputs'
    figure_inputs.mkdir(parents=True,exist_ok=True)
    import shutil
    for name in ('lr_temporal_predictions.csv', 'xgb_temporal_predictions.csv', 'lstm_hobo_importance.csv', 'lstm_hobo_temporal.json'):
        shutil.copyfile(out/name, figure_inputs/name)
    (out/'protocol.json').write_text(json.dumps({'seed':42,'forecast_horizon_days':3,'target_source':'Observed HOBO status at issue date + 3 calendar days; no target imputation','lstm_history':'30 observation records, including rows with missing future targets','lstm_device':'cpu','torch_threads':2},indent=2)+'\n')
    print(json.dumps(r['metrics']))

if __name__=='__main__':main()
