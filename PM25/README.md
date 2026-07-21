# PM25 Submission Package

This folder reorganizes the Hybrid TimeMixer-XGBoost PM2.5 project to match the supplied DAP391m sample structure.

```text
PM25/
├── Dataset/
├── Models and Data Cleaning/
├── APP/
│   └── Eco-Forecast/
├── Report and Presentation/
└── Hybrid_TimeMixer_XGBoost_PM25_Report.pdf
```

## Main folders

- `Dataset/`: station manifest and raw UK-Air PM2.5 station CSV files.
- `Models and Data Cleaning/`: notebooks, scripts, source package, and tests used for cleaning, modelling, and evaluation.
- `APP/Eco-Forecast/`: Streamlit dashboard following the sample app layout (`app.py`, `components/`, `pipeline/`, `visuals/`, `data/`, `results/`, `figures/`).
- `Report and Presentation/`: paper, generated figures/results, and presentation slides.

## Run the app

```powershell
cd "PM25\APP\Eco-Forecast"
streamlit run app.py
```

## Run the full research pipeline

From the original repository root, use:

```powershell
python scripts/run_colab_full_study.py
```

or run the individual scripts inside `Models and Data Cleaning/scripts/`.
