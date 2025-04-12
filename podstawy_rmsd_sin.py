import matplotlib.pyplot as plt
import numpy as np
import random 
##### sinus function with noise generator
# input - vector x with values, and a - noise coefficent
def sin_f(x, a):
	noise = np.random.rand(*x.shape) - 0.5
	return (np.sin(x)) + a*noise
##### generating 3 sine functions
# generating vector x_ from 0 to 6.28 with 20 values between
x_ = np.linspace(0, 12.56, 60)
# generating sin values for input vector x_
Y_sin = sin_f(x_, 0) # 0, so no random noise added
y_sin_small_noise = sin_f(x_, 0.2) # 0.2 so small noise added
y_sin_big_noise = sin_f(x_, 1.5) # 1.5 so big noise added
##### rmse calculation with mean function
# mean - this function calculate average value
# sqrt - this function calculate square root
meanSquaredError_small_noise = ((y_sin_small_noise - Y_sin) ** 2).mean()
rmse_sn = np.sqrt(meanSquaredError_small_noise)
meanSquaredError_big_noise = ((y_sin_big_noise - Y_sin) ** 2).mean()
rmse_bn = np.sqrt(meanSquaredError_big_noise)
print(f'rmse error: small noise={rmse_sn:.3f}, big noise={rmse_bn:.3f}')
##### plots
plt.plot(x_, y_sin_small_noise, linestyle='dotted', c= 'black', label='sine + small noise')
plt.plot(x_, y_sin_big_noise, linestyle='dashed', c= 'blue', label='sine + big noise')
plt.plot(x_, Y_sin, linestyle='solid', c= 'red', label='sine')
plt.text(0, 1.6, f'rmse sn={rmse_sn:.3f}, rmse bn={rmse_bn:.3f}', c= 'black')
plt.xlabel('x')
plt.ylabel('value')
plt.legend()
plt.show()
