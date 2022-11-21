import pulp
from scipy import integrate
import numpy as np
import math
from sympy import symbols, exp, log


def f(t):
    return np.exp(-0.5 * (t ** 2))

# def g(z):
""" Функция стандартного нормального распределения """
# return 1 / np.sqrt(2 * np.pi) * integrate.quad(f, -np.inf, z)


def g(z):
    return exp(z) / (1 + exp(z))


n = 3
m = 3

a = []
for i in range(m):
    a.append(symbols('a' + str(i)))

y = [0, 0, 1]
x = [[3.2, 4.1, 9.0], [1.2, 0.5, 6.0], [9.2, 3.1, 1.2]]

f = 0
for i in range(n):
    z = 0
    for j in range(m):
        z = a[j] * x[i][j]
    f += y[i] * log(g(z)) + (1 - y[i]) * log(1 - g(z))

print(f)
