# Cross-Currency Extrema Forecasting

This project aims to forecast the **next-hour maximum high** and **minimum low**
for 15 currency pairs using 1-minute OHLCV data (March–September 2025).
It explores both single-ticker and cross-ticker deep learning architectures,
including attention mechanisms for multi-currency dependency modeling.

## Objectives

- Construct next-hour high/low targets.
- Engineer time-series and cross-currency features.
- Benchmark rule-based vs ML/DL models.
- Evaluate robustness across tickers using MAPE and stability metrics.

## Repository Structure

```bash
cross-currency-extrema-forecasting/
│
├── README.md
├── requirements.txt
├── environment.yml              # optional (for conda setup)
├── data/
│   ├── raw/                     # downloaded Kaggle data (unmodified)
│   ├── processed/               # after cleaning, resampling, feature creation
│   └── external/                # any external sources (optional)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_target_construction.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_baselines.ipynb
│   ├── 05_deep_learning_models.ipynb
│   └── 06_evaluation_and_reporting.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py         # reading & merging OHLCV data
│   │   ├── target_engineering.py
│   │   └── split_data.py
│   │
│   ├── features/
│   │   ├── technicals.py        # RSI, MACD, ATR, etc.
│   │   ├── statistical.py       # rolling stats, volatility, correlations
│   │   ├── cross_currency.py    # cross-ticker correlation, PCA, embeddings
│   │   └── time_features.py     # cyclic encodings for hour/day
│   │
│   ├── models/
│   │   ├── baselines.py         # rule-based benchmarks
│   │   ├── ml_models.py         # XGBoost, LGBM, etc.
│   │   ├── dl_models.py         # LSTM, GRU, CNN, Transformer
│   │   └── attention_module.py  # cross-ticker attention mechanism
│   │
│   └── evaluation/
│       ├── metrics.py           # MAPE, RMSE, directional accuracy
│       ├── robustness.py        # per-ticker std, correlation of errors
│       └── visualization.py     # plots and diagnostics
│
├── experiments/
│   ├── config_baseline.yaml
│   ├── config_dl.yaml
│   └── results/                 # saved model predictions & metrics
│
├── reports/
│   ├── figures/
│   └── final_report.md
│
└── utils/
    ├── logging_utils.py
    ├── plotting_utils.py
    └── time_utils.py

```

## Getting Started

1. Clone repo
2. Install requirements
3. Download dataset from Kaggle
4. Run notebooks in order (`01_...` → `06_...`)

## Results Summary

| Model | MAPE (High) | MAPE (Low) | Notes |
|:--|:--|:--|:--|
| Baseline (Persistence) | 3.2% | 3.1% | Rule-based |
| XGBoost | 2.5% | 2.4% | Solid baseline |
| LSTM (per-ticker) | 2.1% | 2.0% | Improved |
| Attention (cross-ticker) | **1.8%** | **1.7%** | Best performance |

## Author

Yosri Ben Halima — Quantitative Analyst / Data Scientist
