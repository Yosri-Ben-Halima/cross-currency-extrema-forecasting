# Cross-Currency Extrema Forecasting

This project aims to forecast the **next-hour maximum high** and **minimum low**
for 15 currency pairs using 1-minute OHLCV data (March–September 2025).
It explores both single-ticker and cross-ticker deep learning architectures,
including attention mechanisms for multi-currency dependency modeling.

## Objectives

- Construct next-hour high/low targets and metalabels.
- Engineer time-series and cross-currency features.
- Orthogonalize features and select components based on Max Relevance Min Redundance framework
- Benchmark rule-based vs ML/DL models.
- Evaluate robustness across tickers using MAPE and RMSE.

## Repository Structure

```bash
cross-currency-extrema-forecasting/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── currencies_market_data.parquet
│   │                  
│   ├── processed/
│   │   ├── clean_data.parquet
│   │   ├── labeled_data.parquet
│   │   └── final_data.parquet
│   │            
│   └── split/
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet              
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_target_construction.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_baselines.ipynb
│   ├── 05_deep_learning_models.ipynb
│   └── 06_evaluation_and_reporting.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py         
│   │   ├── target_engineering.py
│   │   └── split_data.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technicals.py        
│   │   ├── returns.py       
│   │   ├── volatility.py       
│   │   ├── microstructure.py     
│   │   ├── cross_currency.py
│   │   ├── time_features.py
│   │   ├── feature_calculator.py
│   │   └── feature_selector.py
│   │
│   ├── models/
│   │   ├── baselines.py         # rule-based benchmarks
│   │   ├── ml_models.py         # XGBoost, LGBM, etc.
│   │   ├── dl_models.py         # LSTM, GRU, CNN, Transformer
│   │   └── attention_module.py  # cross-ticker attention mechanism
│   │
│   └── evaluation/
│       ├── metrics.py           # MAPE, RMSE
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
    ├── helpers.py
    └── viz.py

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
