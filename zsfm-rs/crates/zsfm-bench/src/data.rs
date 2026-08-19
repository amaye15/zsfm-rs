use std::path::Path;

use anyhow::{Context, Result};

pub struct DatasetSpec {
    pub name: &'static str,
    pub file: &'static str,
    pub col: &'static str,
    pub test_rows: usize,
    pub freq: &'static str,
    pub domain: &'static str,
}

/// All 21 datasets, in the order the full report presents them. Mirrors the
/// old `benchmark/run_bench.py`'s `DATASETS` + `gen_report.py`'s
/// `DATASET_META`/`DATASETS_ORDER`, unchanged — same CSV files, same columns,
/// same held-out test-row counts, so results are directly comparable to the
/// pre-rewrite `benchmark.md`.
#[rustfmt::skip]
pub const DATASETS: &[DatasetSpec] = &[
    DatasetSpec { name: "ETTh1", file: "ETTh1.csv", col: "OT", test_rows: 2880, freq: "1h", domain: "Energy" },
    DatasetSpec { name: "ETTh2", file: "ETTh2.csv", col: "OT", test_rows: 2880, freq: "1h", domain: "Energy" },
    DatasetSpec { name: "ETTm1", file: "ETTm1.csv", col: "OT", test_rows: 11520, freq: "15min", domain: "Energy" },
    DatasetSpec { name: "ETTm2", file: "ETTm2.csv", col: "OT", test_rows: 11520, freq: "15min", domain: "Energy" },
    DatasetSpec { name: "electricity", file: "electricity_h1.csv", col: "value", test_rows: 2880, freq: "1h", domain: "Energy" },
    DatasetSpec { name: "solar", file: "solar_1h.csv", col: "value", test_rows: 2000, freq: "1h", domain: "Energy" },
    DatasetSpec { name: "wind", file: "wind_farms_h1.csv", col: "value", test_rows: 2000, freq: "1h", domain: "Energy" },
    DatasetSpec { name: "aus_electricity", file: "aus_electricity_30min.csv", col: "value", test_rows: 20000, freq: "30min", domain: "Energy" },
    DatasetSpec { name: "weather", file: "weather_s1.csv", col: "value", test_rows: 1000, freq: "1h", domain: "Climate" },
    DatasetSpec { name: "weather_10min", file: "weather_10min.csv", col: "OT", test_rows: 10560, freq: "10min", domain: "Climate" },
    DatasetSpec { name: "jena", file: "jena_10min.csv", col: "value", test_rows: 10560, freq: "10min", domain: "Climate" },
    DatasetSpec { name: "melbourne_temp", file: "melbourne_temp.csv", col: "value", test_rows: 365, freq: "1d", domain: "Climate" },
    DatasetSpec { name: "co2", file: "co2_weekly.csv", col: "value", test_rows: 300, freq: "1w", domain: "Climate" },
    DatasetSpec { name: "sunspot_daily", file: "sunspot_daily.csv", col: "value", test_rows: 5000, freq: "1d", domain: "Astronomy" },
    DatasetSpec { name: "sunspot_monthly", file: "sunspot_monthly.csv", col: "value", test_rows: 300, freq: "1mo", domain: "Astronomy" },
    DatasetSpec { name: "ili", file: "ili.csv", col: "OT", test_rows: 200, freq: "1w", domain: "Health" },
    DatasetSpec { name: "exchange", file: "exchange_rate.csv", col: "OT", test_rows: 1500, freq: "1d", domain: "Finance" },
    DatasetSpec { name: "m4_daily", file: "m4_daily.csv", col: "value", test_rows: 2000, freq: "1d", domain: "Finance" },
    DatasetSpec { name: "traffic", file: "traffic_h1.csv", col: "OT", test_rows: 2880, freq: "1h", domain: "Transport" },
    DatasetSpec { name: "pedestrian", file: "pedestrian_counts.csv", col: "value", test_rows: 5000, freq: "1h", domain: "Transport" },
    DatasetSpec { name: "saugeeen", file: "saugeeen_river.csv", col: "value", test_rows: 2000, freq: "1d", domain: "Hydrology" },
];

pub fn find(name: &str) -> Option<&'static DatasetSpec> {
    DATASETS.iter().find(|d| d.name == name)
}

/// Read `col` from `data_dir/<spec.file>` as an f32 series, skipping blank
/// cells (matches the old Python loader's `if row[col].strip()` filter).
pub fn load_series(data_dir: &Path, spec: &DatasetSpec) -> Result<Vec<f32>> {
    let path = data_dir.join(spec.file);
    let mut reader =
        csv::Reader::from_path(&path).with_context(|| format!("open {}", path.display()))?;
    let headers = reader.headers()?.clone();
    let col_idx = headers
        .iter()
        .position(|h| h == spec.col)
        .with_context(|| format!("column {:?} not found in {}", spec.col, path.display()))?;

    let mut series = Vec::new();
    for result in reader.records() {
        let record = result.with_context(|| format!("read row in {}", path.display()))?;
        let Some(cell) = record.get(col_idx) else {
            continue;
        };
        let cell = cell.trim();
        if cell.is_empty() {
            continue;
        }
        series.push(
            cell.parse::<f32>()
                .with_context(|| format!("parse {cell:?} as f32"))?,
        );
    }
    Ok(series)
}
