# Sample Alignment Notes

The supplied `PM25-Weather-DAP391m-Report-Paper-Sample.zip` was used as a
format reference, not as a data source.

## What was adapted

- Dashboard organization: main router plus component tabs.
- Separate dashboard data-loading layer.
- Separate reusable visualization helpers.
- Clear README structure and import map.
- Presentation-friendly Q&A/about pages.

## What was intentionally not copied

- NSW air-quality/weather dataset.
- Sample trained `.pth` and `.json` model files.
- Sample model performance claims.
- Sample station IDs and forecast horizons.

## This project's retained scope

- Dataset: UK-Air PM2.5 station files in `data/raw/stations/`.
- Stations: MY1, BIR_A4540, HP1, CHB, KC1.
- Horizons: 1, 6, 12, 24, and 48 hours.
- Method: gated TimeMixer-XGBoost residual correction.
- Interpretation: partial improvement over TimeMixer, not a SOTA claim.

