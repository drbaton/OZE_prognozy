import numpy as np
import matplotlib as plt
import scipy.optimize as opt
a = np.array([1, 2, 3, 4.1, 5.987, 6])
b = 1.23
for each in a:
    c = each + b
    print(f' liczba {each} plus {b} = {c}')
print('--- koniec ---')
