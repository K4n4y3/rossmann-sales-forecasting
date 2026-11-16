# Data

Source: Kaggle competition **Rossmann Store Sales** — <https://www.kaggle.com/competitions/rossmann-store-sales/data>

The files are not committed to this repository (Kaggle competition rules). Put them in `data/raw/`:

| File | Rows | Content |
|---|---|---|
| `train.csv` | 1,017,209 | daily sales per store, 2013-01-01 to 2015-07-31 |
| `store.csv` | 1,115 | store master data (type, assortment, competition, Promo2) |
| `test.csv` | 41,088 | 856 stores, 2015-08-01 to 2015-09-17, no sales (Kaggle test) |
| `sample_submission.csv` | 41,088 | submission format (`Id`, `Sales`) |

Zipped versions (`train.csv.zip`, …) are unpacked automatically.

## Option A — Kaggle API

```bash
pip install kaggle
# put kaggle.json (Kaggle → Settings → Create New Token) in ~/.kaggle/ and accept the competition rules once on the website
kaggle competitions download -c rossmann-store-sales -p data/raw
cd data/raw && unzip rossmann-store-sales.zip && cd ../..
```

## Option B — manual

Download the four CSV files from the competition's *Data* tab and copy them into `data/raw/`.

## Generated files (ignored by git)

* `data/clean.parquet` — train merged with store, data-quality flags (notebook 01 / `rossmann.data.build_clean`)
* `data/panel.parquet` — clean + test rows with all engineered features (`rossmann.features.build_panel`)
* `data/daily_total.parquet` — daily aggregate series (notebook 01)
