import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
##### import data to X and Y
data = np.genfromtxt('PV_data_3mce.csv', delimiter=';', skip_header=1)
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
def linear(x, a):
    return a * x
##### fitting to function
(a), _ = opt.curve_fit(linear, train_zmienne, train_y, p0=(1))
print(f'a={a[0]}')
##### rmse calculation
Y_pred = linear(test_zmienne, a)
meanSquaredError = ((Y_pred - test_y) ** 2).mean()
rmse = np.sqrt(meanSquaredError)
max_E = max(Y)
print(f'rmse error:{rmse:.3f}, %error(maxP)={(rmse/max_E)*100:.2f}%')
##### plot - function fit
plt.scatter(train_zmienne, train_y, s=20, c= 'gray', marker='x', linewidths=0.5)
plt.scatter(train_zmienne, linear(train_zmienne, a[0]), s=2, c= 'red', marker='o')
plt.xlabel('i [W/m^2]')
plt.ylabel('E [MWh]')
plt.text(max(X)/3, 0, f'a = {a[0]:.2e}', c= 'black')
plt.savefig('PV_linear_fit.png', dpi=300)
plt.show()
##### plot 2 - real vs predicted
fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
real_d = plt.plot(dt[train:], test_y, c= 'gray', label='real data')
pred_d = plt.plot(dt[train:], Y_pred, c= 'red', label='predicted', linestyle = 'dashed')
plt.xlabel('dt [1 hour]')
plt.ylabel('E [MWh]')
plt.title(f'linear rmse={rmse:.3f}, %error for max={(rmse/max_E)*100:.2f}%', c='black')
plt.legend(loc='upper center')
ax.set_ylim([0, 6])
plt.savefig('PV_linear.png', dpi=300)
plt.show()
dane = np.stack((dt[train:], Y_pred), axis=1)
np.savetxt("PV_linear.csv", dane, delimiter=";", header='hour;E_pred', fmt='%.3f')
##### sample forecast predict for next 5 hours
prognoza_i = np.array([0, 0, 100, 200, 500])
fcst_5h = linear(prognoza_i, a)
print(fcst_5h) 