import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# --- Step 1: Load monthly sales data with pandas ---
df = pd.read_csv('monthly_sales.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

print("--- DATA LOADED ---")
print(df.head())

# --- Step 2: Check stationarity and apply differencing if needed ---
result = adfuller(df['Sales'])
print(f'\nADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] > 0.05:
    print("Data is non-stationary. Applying first-order differencing (d=1)...")
    d = 1
else:
    print("Data is stationary (d=0).")
    d = 0

# --- Step 3: Fit ARIMA(p,d,q) model ---
model = ARIMA(df['Sales'], order=(1, d, 1))
model_fit = model.fit()
print("\n--- MODEL FITTED SUCCESSFULLY ---")
print(model_fit.summary().tables[0])

# --- Step 4: Generate forecasts for next 12 months ---
forecast_steps = 12
forecast_result = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast_result.predicted_mean

print("\n--- 12-MONTH FORECAST ---")
print(forecast_mean)

# --- Step 5: Calculate MAPE and plot actual vs forecast ---
actuals = df['Sales']
predictions = model_fit.fittedvalues
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
print(f"\nModel Fit MAPE: {mape:.2f}%")

# Plotting Actual vs Forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Sales'], label='Actual Sales', color='blue')
plt.plot(forecast_mean.index, forecast_mean, label='12-Month Forecast', color='orange', linestyle='--')
plt.title('Monthly Sales Forecast using ARIMA')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()