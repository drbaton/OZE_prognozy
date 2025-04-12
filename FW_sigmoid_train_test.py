import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
##### import data to X and Y
data = np.genfromtxt('FW_data_3mce.csv', delimiter=';', skip_header=1)
X = data[:, 0]
Y = data[:, 1]
dt = data[:, 2]
##### train and test prep.
rows = len(data)
train = int(rows * 0.8)
train_y = Y[:train]
train_zmienne = X[:train]
test_y = Y[train:]
test_zmienne = X[train:]
##### function definition
def logistic(x, c, a, b):
    return c / (1. + np.exp(a * (b - x)))
##### fitting to function
(c, a, b), _ = opt.curve_fit(logistic, train_zmienne, train_y, p0=(max(Y), 1, 1))
print(f'c={c}, a={a}, b={b}')
##### rmse calculation
Y_predicted = (logistic(test_zmienne,c,a,b))
meanSquaredError = ((Y_predicted - test_y) ** 2).mean()
rmse = np.sqrt(meanSquaredError)
max_E = test_y.max()
print(f'rmse error:{rmse:.3f}, %error(maxP)={(rmse/max_E)*100:.2f}%')
##### plot - function fit
plt.scatter(train_zmienne, train_y, s=20, c= 'gray', marker='x', linewidths=0.5)
plt.scatter(train_zmienne, logistic(train_zmienne, c, a, b), s=2, c= 'red', marker='o')
plt.xlabel('v [m/s]')
plt.ylabel('E [MWh]')
plt.text(max(X)/3, 0, f'{c:.3f} / (1 + exp({a:.3f} * ({b:.3f} - v))', c= 'black')
plt.savefig('FW_sigmo_fit.png', dpi=300)
plt.show()
##### plot 2 - real vs predicted
fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
real_d = plt.plot(dt[train:], Y[train:], c= 'gray', label='real data')
pred_d = plt.plot(dt[train:], Y_predicted, c= 'red', label='predicted', linestyle = 'dashed')
plt.xlabel('dt [1 hour]')
plt.ylabel('E [MWh]')
plt.legend()
plt.title(f'sigmoid rmse={rmse:.3f}, %error for max={(rmse/max_E)*100:.2f}%', c='black')
plt.savefig('FW_sigmoid.png', dpi=300)
plt.show()
dane = np.stack((dt[train:], Y_predicted), axis=1)
np.savetxt("FW_sigmoid.csv", dane, delimiter=";", header='hour;E_pred', fmt='%.3f')
##### sample forecast predict for next 5 hours
prognoza_v = np.array([1.09,2.20,3.67,4.43,10.11])
fcst_5h = logistic(prognoza_v, c, a, b)
print(fcst_5h) 