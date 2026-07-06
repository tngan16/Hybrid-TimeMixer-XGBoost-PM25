# Recommended additional PM2.5 stations

Use official UK-Air hourly column-format CSV files. The project reader accepts
both the supplied MY1 `datetime, PM2.5, Status` format and the common UK-Air
`Date, Time, PM2.5, Status` export format.

| Priority | Station | Type | Years suitable for this project | Reason |
|---|---|---|---|---|
| 1 | KC1 — London N. Kensington | Urban Background | 2016–2025 | Long background contrast to roadside MY1 |
| 2 | HP1 — London Honor Oak Park | Urban Background | 2019–2025 | Independent London background validation |
| 3 | CHBO — Chilbolton Observatory | Rural Background | 2016–2025 | Tests rural transfer and lower concentration regime |
| 4 | MAN3 — Manchester Piccadilly | Urban Background | Verify coverage before use | Tests transfer outside London |
| 5 | Birmingham A4540 Roadside (UKA00626) | Urban Traffic | Verify coverage before use | Second roadside setting outside London |

Current confirmatory set in `data/stations.csv`: **MY1 + BIR_A4540 + HP1 + CHB
+ KC1**. This keeps the original MY1 dataset as the internal comparison point
and adds roadside, urban-background and rural-background stations requested in
the reviewer revision.

Official pages:

- https://uk-air.defra.gov.uk/data/flat_files?site_id=KC1
- https://uk-air.defra.gov.uk/data/flat_files?site_id=HP1
- https://uk-air.defra.gov.uk/data/flat_files?site_id=CHBO
- https://uk-air.defra.gov.uk/data/flat_files?site_id=MAN3

Run `python scripts/preprocess_stations.py` after changing this manifest. The
script writes station-specific cleaned datasets under `data/processed/stations/`
and a cross-station quality comparison to
`results/station_data_quality_comparison.csv`.
