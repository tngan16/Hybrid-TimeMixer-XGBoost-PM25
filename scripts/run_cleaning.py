import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from pm25forecast.pipeline import prepare
print(prepare(ROOT/"data/raw/London_Marylebone_PM25.csv",ROOT)[1])
