# Time Series Forecast Web App
### 배포링크: https://timeseriesonlycsv-rskvusanmofjsnxgjvg2xc.streamlit.app/


단변량 CSV 파일을 업로드하면 자동 예측과 평가지표 대시보드를 확인할 수 있는 시계열 분석 웹앱입니다

## Features
- 단변량 CSV 업로드
- 시평(horizon) 조절
- 파일이나 파라미터 변경 시 재예측
- 모델 비교 대시보드
- 평가지표: MAE, RMSE, MAPE, SMAPE, MASE
- 사용 모델: Naive, Holt, Holt-Winters, AutoARIMA

## 샘플 CSV 파일
- sample_data/monthly_demand_sample.csv
- sample_data/daily_traffic_sample.csv
- sample_data/airline_official_sktime.csv (강의에서 사용한 sktime 항공승객 데이터)

## 파일 형식
- date: 날짜/시간 열
- value: 수치형 타깃 열
