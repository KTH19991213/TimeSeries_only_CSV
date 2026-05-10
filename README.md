# Time Series Forecast Web App
### 배포링크: https://timeseriesonlycsv-rskvusanmofjsnxgjvg2xc.streamlit.app/

Upload a univariate CSV and run automatic forecasting with dashboard metrics.

## Run
1. Install dependencies:
   pip install -r requirements.txt
2. Start app:
   streamlit run app.py

## Features
- CSV upload (univariate)
- Forecast horizon control
- Re-forecast when file or parameters change
- Model comparison dashboard
- Metrics: MAE, RMSE, MAPE, SMAPE, MASE
- Models: Naive, Holt, Holt-Winters, AutoARIMA

## Sample CSV files
- sample_data/monthly_demand_sample.csv
- sample_data/daily_traffic_sample.csv
- sample_data/airline_official_sktime.csv (from `sktime.datasets.load_airline` used in lectures)

All files use this schema:
- date: timestamp column
- value: numeric target column
