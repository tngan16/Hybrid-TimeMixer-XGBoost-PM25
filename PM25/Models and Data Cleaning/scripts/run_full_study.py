import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from pm25forecast.config import Config
from pm25forecast.pipeline import prepare,run_baselines,run_experiment
from pm25forecast.viz import make_all
import subprocess,sys as _sys

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--epochs",type=int,default=30); parser.add_argument("--seed",type=int,default=42); args=parser.parse_args(); raw=ROOT/"data/raw/London_Marylebone_PM25.csv"; cfg=Config(epochs=args.epochs,seed=args.seed)
    print("[1/4] Cleaning and audit"); print(prepare(raw,ROOT,cfg)[1]); print("[2/4] Baselines"); print(run_baselines(raw,ROOT,cfg)); print("[3/4] Neural and hybrid experiment"); print(run_experiment(raw,ROOT,cfg)[0]); print("[4/4] Figures"); print(make_all(ROOT)); subprocess.run([_sys.executable,str(ROOT/"scripts/export_latex_results.py")],check=True)
if __name__=="__main__": main()
