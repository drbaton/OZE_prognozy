import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from prophet import Prophet
##### import data to dateframe 
df = pd.read_csv('PV_data_3mce_timeseries_full.csv', delimiter=';', parse_dates=['ds'])
##### prepare forecast
rows = len(df)
train = int(rows * 0.8)
model = Prophet().fit(df[:train])
forecast_period = (rows-train)
future = model.make_future_dataframe(periods=forecast_period, freq='h')
fcst = model.predict(future)
print(df['ds'][1728])
##### calc rmse
meanSquaredError = ((fcst['yhat'][train:] - df['y'][train:]) ** 2).mean()
rmse = np.sqrt(meanSquaredError)
max_E = df['y'][train:].max()
print(f'rmse={rmse:.3f}, %error for max={(rmse/max_E)*100:.2f}%')
##### "night" filter
fcst['yhat'] = np.where(fcst['yhat'] < 0.2*max_E, 0, fcst['yhat'])
##### plot
fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
plt.plot(df['ds'][train:], df['y'][train:], label='real data', c='gray')
plt.plot(fcst['ds'][train:], fcst['yhat'][train:], label='forecast', c='red', linestyle='dashed')
date_form = DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.title(f'Prophet (stat. forecast) rmse={rmse:.3f}, %error for max={(rmse/max_E)*100:.2f}%', c='black')
plt.xticks(rotation=90)
plt.ylabel('E [MWh]')
plt.legend(loc='upper center')
ax.set_ylim([0, 6])
plt.savefig('PV_stat_train_test.png', dpi=300)
plt.show()
dataset = pd.DataFrame({'data': df['ds'][train:], 'energia': fcst['yhat'][train:]})
dataset.to_csv('PV_stat.csv', index=False, sep=';')
