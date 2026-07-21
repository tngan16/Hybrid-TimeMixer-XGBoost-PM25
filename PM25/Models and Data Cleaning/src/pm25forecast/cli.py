from argparse import ArgumentParser
from pathlib import Path
from .config import Config
from .pipeline import prepare,run_baselines,run_experiment
def args():
    root=Path.cwd(); p=ArgumentParser(); p.add_argument("--input",default=root/"data/raw/London_Marylebone_PM25.csv"); p.add_argument("--epochs",type=int,default=30); return root,p.parse_args()
def clean_main():
    r,a=args(); print(prepare(a.input,r,Config(epochs=a.epochs))[1])
def baseline_main():
    r,a=args(); print(run_baselines(a.input,r,Config(epochs=a.epochs)))
def experiment_main():
    r,a=args(); print(run_experiment(a.input,r,Config(epochs=a.epochs)))

def figures_main():
    from .viz import make_all
    print(make_all(Path.cwd()))
