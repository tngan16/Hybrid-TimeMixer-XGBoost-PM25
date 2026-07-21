import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from pm25forecast.config import Config
from pm25forecast.pipeline import run_experiment
p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=30); a=p.parse_args()
print(run_experiment(ROOT/"data/raw/London_Marylebone_PM25.csv",ROOT,Config(epochs=a.epochs)))
