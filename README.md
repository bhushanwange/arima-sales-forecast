# ARIMA Sales Forecast Project

A simple machine learning time-series forecasting project that uses an ARIMA model to analyze monthly sales and generate future predictions.

## Files in this Project
* **`monthly_sales.csv`**: Contains historical monthly sales data.
* **`arima_model.py`**: The main Python script that loads data, checks stationarity, fits the ARIMA model, forecasts the next 12 months, and plots the results.

## Requirements & Dependencies
Make sure you have Python installed along with the required libraries:
```bash
pip install pandas numpy matplotlib statsmodels