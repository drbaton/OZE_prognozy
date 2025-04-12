import numpy as np
import pandas as pd
X = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([3, 2.8, 2.6, 3.2, 2.9, 3.1])
# numpy file saving
dane = np.stack((X, Y), axis=1)
np.savetxt("dane_np.csv", dane, delimiter=";", header='x;y', fmt='%.3f')
# pandas file saving
dataset = pd.DataFrame({'xval': X, 'yval': Y})
dataset.to_csv('dane_pandas.csv', index=False, sep=';')