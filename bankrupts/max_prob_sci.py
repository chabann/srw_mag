from scipy import optimize, integrate
import numpy as np
import math
from preprocessing_data import preprocess
import pandas as pd


def g_logit(z):
    return np.exp(z) / (1 + np.exp(z))


def f1(t):
    return np.exp(-0.5 * (t ** 2))


def g_probit(z):
    return 1 / np.sqrt(2 * np.pi) * integrate.quad(f1, -np.inf, z)


def print_callback(x, f, accepted):
    print("at minimum %.4f accepted %d values:" % (f, int(accepted)))
    print(x)


def f(a):
    val = 0
    for i in range(n):
        z = 0
        for j in range(m):
            z += a[j] * x[i][j]
        if y[i] == 1:
            val += y[i] * math.log(g_probit(z))
        else:
            val += (1 - y[i]) * math.log(1 - g_probit(z))
    return -val


def f2(a):
    val = 0
    for i in range(n):
        z = 0
        for j in range(m):
            z = a[j] * x[i][j]
        val += y[i] - z
    return val


# fileName = '../data/financeData.csv'
# df = pd.read_csv(fileName)
df = preprocess('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data.csv')

m = 4

# bankrupts = df['bankrupt']

y = []
x = []

n = len(y)

maxi = 4000
for index, row in df.iterrows():
    if index > maxi:
        break
    # params = [(row['current1200'] - row['current1500']) / row['current1600'], row['current1370'] / row['current1600'],
              # row['current2300'] / row['current1600'], row['current1300'] / (row['current1400'] + row['current1500'])]

    params = [(row['1200 (2018)'] - row['1500 (2018)']) / row['1600 (2018)'], row['1370 (2018)'] / row['1600 (2018)'],
              row['2300 (2018)'] / row['1600 (2018)'], row['1300 (2018)'] / (row['1400 (2018)'] + row['1500 (2018)'])]

    """for j in range(m):
        if (params[j] is None) or not (params[j] >= 0 or params[j] <= 0):
            params[j] = 0
        elif np.abs(params[j] > 100000):
            params[j] /= 1000"""
    if np.all(np.isfinite(params)) and np.all(params != 0):
        x.append(params)
        y.append(row['Label'])

# y = [0, 0, 1]
# x = [[1.2, 4.1, 9.0, 6.9], [1.2, 0.5, 3.0, 0.8], [0.2, 3.1, 1.2, 0.7]]

# a0 = np.array([np.random.randn() for i in range(m)])
a0 = np.array([0 for i in range(m)])

rng = np.random.default_rng()
bounds = [(-30, 30) for _ in range(m)]
res = optimize.minimize(f, a0, method='Nelder-Mead', options={'xtol': 1e-12, 'disp': True})
# ret = optimize.basinhopping(f, a0, callback=print_callback, seed=rng)
# result = optimize.differential_evolution(f, bounds)
# nelder-mead
# powell
print('')
print('Optimization result:')
print(res.x, res.fun)
# print(ret.x)
# print(result.x, result.fun)


# Create an object with Poisson model values
# poi = PoissonRegression(np.array(yi), np.array(xi), a0)
# Use newton_raphson to find the MLE
# ares = newton_raphson(poi, display=True)


"""def newton_raphson(model, tol=1e-3, max_iter=1000, display=True):

    i = 0
    error = 100  # Initial error value

    # Print header of output
    if display:
        header = f'{"Iteration_k":<13}{"Log-likelihood":<16}{"theta":<60}'
        print(header)
        print("-" * len(header))

    # While loop runs while any value in error is greater
    # than the tolerance until max iterations are reached
    while np.any(error > tol) and i < max_iter:
        H, G = model.H(), model.G()
        beta_new = model.beta - (np.linalg.inv(H) @ G)
        error = beta_new - model.beta
        model.beta = beta_new

        # Print iterations
        if display:
            beta_list = [f'{t:.3}' for t in list(model.beta.flatten())]
            update = f'{i:<13}{model.logL():<16.8}{beta_list}'
            print(update)

        i += 1

    print(f'Number of iterations: {i}')
    print(f'beta_hat = {model.beta.flatten()}')

    # Return a flat array for beta (instead of a k_by_1 column vector)
    return model.beta.flatten()

class PoissonRegression:

    def __init__(self, y, X, beta):
        self.X = X
        self.n, self.k = X.shape
        # Reshape y as a n_by_1 column vector
        self.y = y.reshape(self.n, 1)
        # Reshape beta as a k_by_1 column vector
        self.beta = beta.reshape(self.k, 1)

    def mu(self):
        return np.exp(self.X @ self.beta)

    def logL(self):
        y = self.y
        mu = self.mu()
        return np.sum(y * np.log(mu) - mu - np.log(math.factorial(y)))

    def G(self):
        y = self.y
        mu = self.mu()
        X = self.X
        return X.T @ (y - mu)

    def H(self):
        X = self.X
        mu = self.mu()
        return -(X.T @ (mu * X))"""