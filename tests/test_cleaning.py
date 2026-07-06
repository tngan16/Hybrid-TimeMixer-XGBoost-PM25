import numpy as np
import pandas as pd
from pm25forecast.io import read_uk_air_csv
from pm25forecast.quality import annotate_measurements, flag_outliers
from pm25forecast.cleaning import complete_hourly_grid, interpolate_short_internal_gaps, gap_table

def raw_frame():
    return pd.DataFrame({
        "datetime":pd.to_datetime(["2024-01-01 00:00Z","2024-01-01 01:00Z",
                                   "2024-01-01 03:00Z","2024-01-01 04:00Z",
                                   "2024-01-01 05:00Z","2024-01-01 06:00Z",
                                   "2024-01-01 07:00Z","2024-01-01 08:00Z"]),
        "PM2.5":[10,-2,14,np.nan,np.nan,np.nan,np.nan,18],
        "Status":["V ugm-3 (TEOM FDMS)"]*4+["V ugm-3 (BAM)"]*4,
        "source_row":range(2,10),"source_file":"synthetic.csv",
    })

def test_quality_flags_and_instrument_encoding():
    x=annotate_measurements(raw_frame())
    assert x.negative_pm25.sum()==1
    assert x.missing_or_non_numeric.sum()==4
    assert set(x.instrument_type.dropna().astype(int))=={0,1}
    assert x.instrument_change.sum()==1
    assert np.isnan(x.loc[x.negative_pm25,"pm25_observed"]).all()

def test_grid_and_short_gap_policy_preserves_long_gap():
    x=interpolate_short_internal_gaps(complete_hourly_grid(annotate_measurements(raw_frame())),3)
    assert len(x)==9                         # missing timestamp at 02:00 inserted
    assert x.inserted_grid_row.sum()==1
    # 01:00 and 02:00 form a two-hour bounded gap and are interpolated.
    assert x.loc[x.datetime.dt.hour.isin([1,2]),"was_interpolated"].all()
    # The four-hour 04:00--07:00 gap remains missing.
    long=x.datetime.dt.hour.isin([4,5,6,7])
    assert x.loc[long,"pm25_clean"].isna().all()
    assert x.loc[long,"long_gap_preserved"].all()
    gaps=gap_table(x)
    assert gaps.hours.tolist()==[4,2]

def test_reader_accepts_date_time_schema(tmp_path):
    path=tmp_path/"station.csv"
    path.write_text(
        "Date,Time,PM2.5,Status\n"
        "2021-01-01,01:00:00,36.816,V ugm-3 (Ref.eq)\n",
        encoding="utf-8",
    )
    x=annotate_measurements(read_uk_air_csv(path))
    assert x.datetime.iloc[0] == pd.Timestamp("2021-01-01 01:00:00+00:00")
    assert int(x.instrument_type.iloc[0]) == 2

def test_outlier_flags_are_preserved_as_screening_metadata():
    x=flag_outliers(interpolate_short_internal_gaps(complete_hourly_grid(annotate_measurements(
        pd.DataFrame({
            "datetime":pd.date_range("2024-01-01",periods=40,freq="h",tz="UTC"),
            "PM2.5":[10.0]*39+[200.0],
            "Status":["V ugm-3 (Ref.eq)"]*40,
            "source_row":range(2,42),
            "source_file":"synthetic.csv",
        })
    )),3))
    assert x.pm25_clean.iloc[-1] == 200.0
    assert x.outlier_iqr_high.iloc[-1]
