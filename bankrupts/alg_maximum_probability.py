import scipy.stats as sps
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy import optimize, integrate


def g_logit(z):
    return np.exp(z) / (1 + np.exp(z))


def f1(t):
    return np.exp(-0.5 * (t ** 2))


def g_probit(z):
    value = 1 / np.sqrt(2 * np.pi) * integrate.quad(f1, -np.inf, z)[0]
    return value


def f_probit(theta):
    val = 0
    for i in range(n):
        yiL = 0
        for j in range(n2):
            yiL += theta[j] * xi[i][j]
        if yi[i] == 1:
            val += yi[i] * math.log(g_probit(yiL))
        else:
            val += (1 - yi[i]) * math.log(1 - g_probit(yiL))
    return -val


def f_logit(theta):
    val = 0
    for i in range(n):
        yiL = 0
        for j in range(n2):
            yiL += theta[j] * xi[i][j]
        if yi[i] == 1:
            val += yi[i] * math.log(g_logit(yiL))
        else:
            val += (1 - yi[i]) * math.log(1 - g_logit(yiL))
    return -val


n = 1000  # кол-во экземпляров
n2 = 2
m = 20  # кол-во подинтервалов
positiveCount = 510  # кол-во положительных

xi = sps.norm(loc=1100, scale=300).rvs(size=n)
xi.sort()

# xtemp = np.linspace(-1, 1, n)
# p_yi = 1 / (1 + np.exp(-3 * xtemp))

p_yi = sps.norm(loc=1100, scale=300).cdf(xi)

yi = []
for i in range(n):
    if random.random() <= p_yi[i]:
        yi.append(1)
    else:
        yi.append(0)

l = (xi[-1] - xi[0]) / m
p = positiveCount/m

current = xi[0]
intervals = [current]
centers = []
for _ in range(m):
    current += l
    intervals.append(current)
    centers.append(current - l / 2)

lastIndex = 0
arCount = []
arValues = []
arPosCount = []
for i in range(m):
    count = 0
    countPositive = 0
    while xi[lastIndex] <= intervals[i + 1]:
        count += 1
        if yi[lastIndex] == 1:
            countPositive += 1

        if (lastIndex + 1) < n:
            lastIndex += 1
        else:
            break
    arCount.append(count)
    arPosCount.append(countPositive)
    if count:
        arValues.append(countPositive / count)
    else:
        arValues.append(0)

print(sum(yi))

plt.figure(1)
plt.plot(centers, arValues, '.r--', label='G_n')
plt.grid()

"""
plt.figure(2)
plt.plot(centers, arPosCount, 'b.')
plt.grid()
"""

xi = np.array(xi)
xi.shape = (n, 1)
e = np.ones((n, 1))
xi = np.append(xi, e, axis=1)

a0 = np.array([0 for i in range(2)])
res_probit = optimize.minimize(f_probit, a0, method='Nelder-Mead', options={'xtol': 1e-9, 'disp': True})
print('Probit:')
print(res_probit.x)
theta_probit = res_probit.x

mu_probit = - theta_probit[1] / theta_probit[0]
sigma_probit = 1 / theta_probit[0]

print(f'mu: {mu_probit} sigma: {sigma_probit}')

# methods = Powell / Nelder-Mead

yi_probit = []
for i in range(n):
    yi_probit.append(g_probit(theta_probit[0] * xi[i][0] + theta_probit[1] * xi[i][1]))


res_logit = optimize.minimize(f_logit, a0, method='Nelder-Mead', options={'xtol': 1e-9, 'disp': True})
print('Logit:')
print(res_logit.x)
theta_logit = res_logit.x

mu_logit = - theta_logit[1] / theta_logit[0]
sigma_logit = 1 / theta_logit[0]

print(f'mu: {mu_logit} sigma: {sigma_logit}')

yi_logit = []
for i in range(n):
    yi_logit.append(g_logit(theta_logit[0] * xi[i][0] + theta_logit[1] * xi[i][1]))

theretical = sps.norm(loc=1100, scale=300).cdf(centers)
logit = sps.norm(loc=mu_logit, scale=sigma_logit).cdf(centers)
probit = sps.norm(loc=mu_probit, scale=sigma_probit).cdf(centers)

plt.plot(centers, theretical, 'y-', label='Theoretical')
plt.plot(centers, logit, 'b-', label='Logit-model')
plt.plot(centers, probit, 'g-', label='Probit-model')

plt.legend()
plt.show()