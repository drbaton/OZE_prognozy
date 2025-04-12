import numpy as np
import matplotlib as plt
import scipy.optimize as opt
a = np.array([1, 2, 3, 4.1, 5.987, 6])
x = 1.23
def kwadratowa(x, a):
    return a * x **2
c = kwadratowa(x, a)
print(f' wartość f. kwadratowej = {c}')
print(f' wymiar zmiennej c = {c.shape}')
print('--- koniec ---')
