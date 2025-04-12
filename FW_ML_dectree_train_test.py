import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from sklearn.tree import DecisionTreeRegressor
##### import data to dateframe 
df = pd.read_csv('FW_data_3mce_timeseries_full.csv', delimiter=';', decimal='.', parse_dates=['ds'])
##### train and test prep.
rows = len(df)
train = int(rows * 0.8)
df_train_y = df['P'][:train]
df_train_zmienne = df[:train].drop(['ds', 'dt', 'P'], axis = 1)
df_test_y = df['P'][train:]
df_test_zmienne = df[train:].drop(['ds', 'dt', 'P'], axis = 1)
##### fit and predict for test
model = DecisionTreeRegressor(max_depth=4)
model.fit(df_train_zmienne, df_train_y)
fcst = model.predict(df_test_zmienne)
##### calc rmse
meanSquaredError = ((fcst - df_test_y) ** 2).mean()
rmse = np.sqrt(meanSquaredError)
max_E = df_test_y.max()
print(f'rmse={rmse:.3f}, %error for max={(rmse/max_E)*100:.2f}%')
##### plot train, test and predict
fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
plt.plot(df['ds'][train:], df['P'][train:], label='test', c='gray')
plt.plot(df['ds'][train:], fcst, label='predicted', c='red', linestyle = 'dashed')
date_form = DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.xticks(rotation=90, fontsize=8)
plt.title(f'ML DecisionTree rmse={rmse:.3f}, %error for max={(rmse/max_E)*100:.2f}%', c='black')
plt.ylabel('E [MWh]')
plt.legend(loc='upper center')
plt.savefig('FW_ML_dectree_test_train.png')
plt.show()
##### save data
dataset = pd.DataFrame({'data': df['ds'][train:], 'energia': fcst})
dataset.to_csv('FW_ML_dectree.csv', index=False, sep=';')
##### sample forecast predict for next 5 hours
prognoza_v = pd.DataFrame({"V": [1.09,2.20,3.67,4.43,10.11]})
fcst_5h = model.predict(prognoza_v)
print(fcst_5h) 
