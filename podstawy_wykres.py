import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
a = 1.23456
def kwadratowa(x, a):
    return a * x ** 2
y = kwadratowa(x, a)
plt.scatter(x, y, s=50, c='red')
plt.plot(x, y, c='blue')
plt.text(min(x), min(y), f'wykres f.kwadratowej dla a={a:.1f}', c='black')
plt.xlabel('oś x')
plt.ylabel('oś y')
plt.show()
