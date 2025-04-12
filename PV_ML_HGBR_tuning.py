import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from sklearn.ensemble import HistGradientBoostingRegressor
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
parametr_max_depth = np.arange(1, 21, 1)
parametr_max_leaf_nodes = np.arange(2, 12, 1)
##### creating Yrmse with zeros 
Yrmse = np.zeros((21, 12))
for each_md in parametr_max_depth:
	for each_mln in parametr_max_leaf_nodes:
		model = HistGradientBoostingRegressor(max_depth=each_md, max_leaf_nodes=each_mln)
		model.fit(df_train_zmienne, df_train_y)
		fcst = model.predict(df_test_zmienne)
		##### calc rmse
		meanSquaredError = ((fcst - df_test_y) ** 2).mean()
		rmse = np.sqrt(meanSquaredError)
		max_E = df_test_y.max()
		print(f' - parametr md={each_md}, parametr mln={each_mln}, rmse={rmse:.3f}, %error for max={(rmse/max_E)*100:.2f}%')
		##### adding values to Yrmse
		Yrmse[each_md][each_mln] = rmse

Yrmse = np.where(Yrmse==0, np.nan, Yrmse)
fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
plt.imshow(Yrmse, cmap='jet', aspect='auto')  
plt.title( 'rmse map for HGBR', fontsize=12)
plt.xlabel('max_leaf_nodes', fontsize=10)
plt.ylabel('max_depth', fontsize=10)
plt.xticks(parametr_max_leaf_nodes, fontsize=8)
plt.yticks(parametr_max_depth, fontsize=8)
plt.colorbar()
plt.show()
