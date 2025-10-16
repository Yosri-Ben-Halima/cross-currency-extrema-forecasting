# Cross-Currency Extrema Forecasting

This project aims to forecast the **next-hour maximum high** and **minimum low**
for 15 currency pairs using 1-minute OHLCV data.

## Objectives

- Construct next-hour max high/min low targets and metalabels.
- Engineer features.
- Stationarize and orthogonalize features and select components based on Max Relevance Min Redundance framework
- Benchmark rule-based vs XGBoost Regression models and evaluate preformance across tickers using MAPE and RMSE.

## Repository Structure

```bash
cross-currency-extrema-forecasting/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
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
│   └── 04_modeling_and_benchmarking.ipynb
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
│   │   ├── baselines.py        
│   │   ├── ml_models.py         
│   │   └── dl_models.py         
│   │
│   └── evaluation/
│       └── metrics.py
│
│
└── utils/
    ├── helpers.py
    └── viz.py
```

## Getting Started

1. Clone repo
2. Install requirements
3. Run notebooks in order (`01_...` → `04_...`)

## Author

Yosri Ben Halima - Quantitative Analyst / Data Scientist
