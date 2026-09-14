"""Regenerate the canonical all-sites LSTM importance panel."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/paper"

def importance():
    import torch
    from lstm_all_sites import sequences
    from hja.data import scale_train_only,BINARY_COLS
    from hja.models.lstm import LSTMModel
    from hja.importance import importance_frame,plot_top10
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

if __name__=="__main__":importance()
