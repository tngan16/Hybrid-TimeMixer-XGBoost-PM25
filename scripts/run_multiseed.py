import argparse,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from pm25forecast.config import Config
from pm25forecast.pipeline import run_experiment

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4]); parser.add_argument("--epochs",type=int,default=30); args=parser.parse_args(); rows=[]
    for seed in args.seeds:
        metrics,_=run_experiment(ROOT/"data/raw/London_Marylebone_PM25.csv",ROOT,Config(seed=seed,epochs=args.epochs)); rows.append(metrics[metrics.model.isin(["dlinear","lstm","timemixer","hybrid_static","hybrid"])].assign(seed=seed))
    frame=pd.concat(rows,ignore_index=True); frame.to_csv(ROOT/"results/seed_metrics.csv",index=False); summary=frame.groupby(["model","horizon"])[["MAE","RMSE","sMAPE","R2"]].agg(["mean","std"]); summary.to_csv(ROOT/"results/multiseed_summary.csv"); print(summary)
if __name__=="__main__": main()
