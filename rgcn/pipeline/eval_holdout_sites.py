"""eval_holdout_sites.py — score the with-sensor site holdout (spatial transfer).

For models trained with ``site_holdout`` (held-out reaches keep their obs-lag
INPUTS but contribute no labels to the loss), this scores their exported
predictions on the held-out reaches' labels — the RGCN analog of the paper's
LR/XGBoost site-based split: a spatially new site *with* an observation
stream. Reports each held-out reach and the pooled set, per horizon and
pooled, on (a) all labeled dates and (b) only dates inside the split's val
blocks (the leakage-guarded subset comparable to the main val metrics).

Run:  RGCN_CONFIG=rgcn/config_consistph_no7_sh.yml uv run python -m rgcn.pipeline.eval_holdout_sites
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .config import load_config


def _metrics(s: pd.DataFrame) -> dict:
    y = s["true_wetdry"].round().astype(int).to_numpy()
    p = s["pred_wetdry_prob"].to_numpy()
    yhat = s["pred_wetdry_label"].to_numpy()
    out = {
        "N": len(s),
        "WetFrac": float(y.mean()) if len(s) else float("nan"),
        "Accuracy": accuracy_score(y, yhat) if len(s) else float("nan"),
        "F1": f1_score(y, yhat, zero_division=0) if len(s) else float("nan"),
        "ROC-AUC": (roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")),
    }
    out["DryRecall"] = (float(((y == 0) & (yhat == 0)).sum() / max((y == 0).sum(), 1))
                        if (y == 0).any() else float("nan"))
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", default=None,
                    help="Override paths.predictions_dir (e.g. a _stride1 export).")
    ap.add_argument("--horizons", default="1,2,3",
                    help="Comma-separated horizon steps to score (e.g. '3' for a "
                         "t+3-only comparison against single-horizon baselines).")
    args = ap.parse_args()

    config = load_config()
    holdout = [int(x) for x in (config.get("site_holdout") or {}).get("nhd_ids", [])]
    if not holdout:
        raise SystemExit("Config has no site_holdout.nhd_ids — nothing to score.")
    pred_dir = (config.repo_root / args.predictions_dir if args.predictions_dir
                else config.path("predictions_dir"))
    horizons = [int(x) for x in args.horizons.split(",")]

    # Val-block membership by DATE (not split-map window ids, which don't apply
    # to eval-stride exports): a prediction date is "val" if it falls inside a
    # holdout block of the blocked split.
    blocks = [(pd.Timestamp(s), pd.Timestamp(e))
              for s, e in config["split"].get("holdout_blocks", [])]

    frames = []
    for h in horizons:
        df = pd.read_csv(pred_dir / f"train_val_predictions_day{h}.csv",
                         usecols=["site_id", "date", "has_true_label",
                                  "true_wetdry", "pred_wetdry_prob", "pred_wetdry_label"],
                         parse_dates=["date"])
        df = df[df["has_true_label"] & df["site_id"].isin(holdout)]
        # Eval-stride grids can predict the same (site, date) from several
        # windows at the same horizon; average the probability.
        df = (df.groupby(["site_id", "date"], as_index=False)
                .agg(true_wetdry=("true_wetdry", "first"),
                     pred_wetdry_prob=("pred_wetdry_prob", "mean")))
        df["pred_wetdry_label"] = (df["pred_wetdry_prob"] >= 0.5).astype(int)
        df["horizon"] = f"Day {h}"
        frames.append(df)
    all_h = pd.concat(frames, ignore_index=True)
    in_block = pd.Series(False, index=all_h.index)
    for s, e in blocks:
        in_block |= (all_h["date"] >= s) & (all_h["date"] <= e)
    all_h["split"] = "other"
    all_h.loc[in_block, "split"] = "val"

    lines = ["# RGCN with-sensor site-holdout evaluation", ""]
    lines.append(f"Checkpoint: `{config['paths']['checkpoint']}` | "
                 f"predictions: `{config['paths']['predictions_dir']}`")
    lines.append("")
    lines.append(f"Held-out reaches (labels masked from training loss; obs-lag "
                 f"inputs intact): {holdout}")
    lines.append("")

    for scope_name, scope in (("All labeled dates", all_h),
                              ("Val-block dates only", all_h[all_h["split"] == "val"])):
        lines.append(f"## {scope_name}")
        lines.append("")
        lines.append("| Slice | N | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |")
        lines.append("|---|--:|--:|--:|--:|--:|--:|")
        rows = [("Pooled (all horizons)", scope)]
        rows += [(f"Day {h}", scope[scope["horizon"] == f"Day {h}"]) for h in horizons]
        rows += [(f"site {sid}", scope[scope["site_id"] == sid]) for sid in holdout]
        for name, s in rows:
            if len(s) == 0:
                continue
            m = _metrics(s)
            lines.append(f"| {name} | {m['N']:,} | {m['WetFrac']:.2f} | "
                         f"{m['Accuracy']:.3f} | {m['ROC-AUC']:.3f} | {m['F1']:.3f} | "
                         f"{m['DryRecall']:.3f} |")
        lines.append("")

    lines.append("Context: 'All labeled dates' overlaps the training period of "
                 "*other* sites (standard site-based-split semantics — same era, "
                 "new site). 'Val-block dates' additionally avoids any temporal "
                 "overlap with training targets.")

    tag = config.path("checkpoint").stem.replace("best_model_", "")
    if args.predictions_dir and "_stride" in args.predictions_dir:
        tag += "_" + args.predictions_dir.rsplit("_", 1)[-1].rstrip("/")
    if horizons != [1, 2, 3]:
        tag += "_h" + "".join(map(str, horizons))
    out = config.repo_root / config["paths"].get(
        "holdout_report", f"results/rgcn_eval_siteholdout_{tag}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
