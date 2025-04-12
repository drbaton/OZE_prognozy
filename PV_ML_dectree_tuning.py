import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from sklearn.tree import DecisionTreeRegressor
##### import data to dateframe 
df = pd.read_csv('PV_data_3mce_timeseries_full.csv', delimiter=';', parse_dates=['ds'])
##### train and test prep.
rows = len(df)
train = int(rows * 0.8)
df_train_y = df['y'][:train]
df_train_zmienne = df[:train].drop(['ds', 'dt', 'y'], axis = 1)
df_test_y = df['y'][train:]
df_test_zmienne = df[train:].drop(['ds', 'dt','y'], axis = 1)
##### fit and predict for test
parametr = np.arange(1, 11, 1)
Yrmse = []
for each in parametr:
	model = DecisionTreeRegressor(max_depth=each)
	model.fit(df_train_zmienne, df_train_y)
	fcst = model.predict(df_test_zmienne)
	##### calc rmse
	meanSquaredError = ((fcst - df_test_y) ** 2).mean()
	rmse = np.sqrt(meanSquaredError)
	max_E = df_test_y.max()
	print(f' - parametr={each}, rmse={rmse:.3f}, %error for max={(rmse/max_E)*100:.2f}%')
	Yrmse.append(rmse)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
plt.plot(parametr, Yrmse)  
plt.title( "rmse for Decision Tree", fontsize=12)
plt.xlabel("max_depth", fontsize=10)
plt.ylabel("rmse", fontsize=10)
plt.show()
