"""Regenerate the raw seed-42 copula paper table and figures, without training."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hja.paths import REPO
from hja.paper_results import raw_copula_table,plot_raw_intervals,plot_raw_map


def main():
    out=REPO/'results/canonical_availability/paper'
    out.mkdir(parents=True,exist_ok=True)
    df=raw_copula_table()
    df.to_csv(out/'raw_seed42_copula.csv',index=False)
    for name,plot in [('raw_seed42_intervals',plot_raw_intervals),('raw_seed42_map',plot_raw_map)]:
        fig=plot(df)
        for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    print(df.to_string(index=False))
    print(f'Included: {df.included.sum()}/{len(df)}; mean width: {(df.hi-df.lo).mean():.3f} days')


if __name__=='__main__':main()
