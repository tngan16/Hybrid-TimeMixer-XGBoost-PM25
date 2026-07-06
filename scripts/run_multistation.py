"""Run the pre-specified 5-seed/30-epoch protocol for user-supplied stations."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pm25forecast.config import Config
from pm25forecast.pipeline import run_experiment


def effectiveness(frame):
    means=(frame.groupby(["station_id","site_type","model","horizon"],
                         as_index=False).MAE.mean())
    comparator=means.loc[~means.model.isin(["hybrid","hybrid_static"])]
    index=comparator.groupby(["station_id","horizon"]).MAE.idxmin()
    best=comparator.loc[index,["station_id","horizon","model","MAE"]].rename(
        columns={"model":"best_comparator","MAE":"comparator_mae"}
    )
    hybrid=means.loc[means.model.eq("hybrid"),
                     ["station_id","site_type","horizon","MAE"]].rename(
        columns={"MAE":"hybrid_mae"}
    )
    out=hybrid.merge(best,on=["station_id","horizon"],how="left")
    out["mae_delta"]=out.hybrid_mae-out.comparator_mae
    out["improvement_percent"]=(
        100*(out.comparator_mae-out.hybrid_mae)/out.comparator_mae
    )
    out["hybrid_wins"]=out.mae_delta.lt(0)
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",default="data/stations.csv")
    parser.add_argument("--epochs",type=int,default=30)
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4])
    parser.add_argument(
        "--train-start",default=None,
        help="Optional common training-history start, e.g. 2021-01-01 00:00:00+00:00"
    )
    parser.add_argument(
        "--output-prefix",default="",
        help="Prefix for sensitivity outputs, e.g. common_2021_"
    )
    args=parser.parse_args()
    manifest=ROOT/args.manifest
    if not manifest.exists():
        raise FileNotFoundError(
            "Create data/stations.csv from data/stations.example.csv after "
            "downloading the additional station datasets."
        )
    stations=pd.read_csv(manifest)
    required={"station_id","station_name","site_type","path"}
    if required-set(stations):
        raise ValueError(f"Manifest missing columns: {sorted(required-set(stations))}")
    rows=[]; bootstrap_rows=[]; ablation_rows=[]; gate_rows=[]; log=[]
    for station in stations.itertuples(index=False):
        raw=ROOT/station.path
        if not raw.exists():
            raise FileNotFoundError(f"Missing dataset for {station.station_id}: {raw}")
        for seed in args.seeds:
            if args.output_prefix:
                artifact_root=(
                    ROOT/"artifacts"/args.output_prefix.rstrip("_")/"stations"
                )
            else:
                artifact_root=ROOT/"artifacts/stations"
            run_root=artifact_root/station.station_id/f"seed_{seed}"
            run_root.mkdir(parents=True,exist_ok=True)
            metrics,diagnostics=run_experiment(
                raw,run_root,Config(
                    seed=seed,epochs=args.epochs,train_start=args.train_start
                )
            )
            rows.append(metrics.assign(
                station_id=station.station_id,
                station_name=station.station_name,
                site_type=station.site_type,
                seed=seed,epochs=args.epochs,
                train_start=args.train_start or "station_available_history",
                method_version="gated_residual_v2",
            ))
            bootstrap_rows.append(pd.DataFrame(
                diagnostics["bootstrap"]
            ).assign(station_id=station.station_id,seed=seed))
            ablation_path=run_root/"results/ablation_metrics.csv"
            ablation_rows.append(pd.read_csv(ablation_path).assign(
                station_id=station.station_id,seed=seed
            ))
            gate_rows.append(pd.read_csv(
                run_root/"results/residual_gate.csv"
            ).assign(station_id=station.station_id,seed=seed))
            log.append({"station_id":station.station_id,"seed":seed,
                        "epochs":args.epochs,
                        "train_start":args.train_start,
                        "output_prefix":args.output_prefix,
                        "method_version":"gated_residual_v2",
                        "status":"completed"})
    results=ROOT/"results"; results.mkdir(exist_ok=True)
    frame=pd.concat(rows,ignore_index=True)
    prefix=args.output_prefix
    frame.to_csv(results/f"{prefix}multistation_seed_metrics.csv",index=False)
    frame.groupby(
        ["station_id","station_name","site_type","model","horizon"]
    )[["MAE","RMSE","sMAPE","R2"]].agg(["mean","std"]).to_csv(
        results/f"{prefix}multistation_summary.csv"
    )
    effectiveness(frame).to_csv(
        results/f"{prefix}station_effectiveness.csv",index=False
    )
    pd.concat(bootstrap_rows,ignore_index=True).to_csv(
        results/f"{prefix}multistation_bootstrap.csv",index=False
    )
    pd.concat(ablation_rows,ignore_index=True).to_csv(
        results/f"{prefix}multistation_ablation.csv",index=False
    )
    pd.concat(gate_rows,ignore_index=True).to_csv(
        results/f"{prefix}multistation_residual_gate.csv",index=False
    )
    (results/f"{prefix}multistation_run_log.json").write_text(
        json.dumps(log,indent=2),encoding="utf-8"
    )


if __name__=="__main__":
    main()
