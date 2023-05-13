import scipy.stats as sps
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy import optimize, integrate
from bankrupts.max_prob.errors import Errors


class AlgorithmMaximumProbability:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.example_number = self.data.shape[0]
        self.coefficient_number = self.data.shape[1]
        self.a0 = np.array([0 for i in range(self.coefficient_number)])
        self.res_logit = {'x': [], 'fun': 0}
        self.res_probit = {'x': [], 'fun': 0}
        self.yi = {'logit': [], 'probit': []}

    def process_logit(self, method):
        """
        Метод инициирует построение регрессионной модели с использованием Логит-модели
        """

        res_logit = optimize.minimize(self.f_logit, self.a0, method=method,
                                      tol=1e-6, options={'disp': True, 'maxiter': 500})

        # print('Logit:')
        # print(res_logit.x)

        self.res_logit = res_logit

        return res_logit

    def get_result(self, method_type):
        if method_type == 'logit':
            res = self.res_logit
        else:
            res = self.res_probit

        theta = res.x
        mu = - theta[1] / theta[0]
        sigma = 1 / theta[0]

        # print(f'{method_type} mu: {mu} sigma: {sigma}')

        yi = []
        for i in range(self.example_number):
            val = self.g_logit(sum([theta[j] * self.data[i][j] for j in range(self.coefficient_number)]))
            if val > 0.5:
                yi.append(1)
            else:
                yi.append(0)
        print(f'yi {method_type}: {yi}')
        print(f'labels {method_type}: {self.labels.tolist()}')

        r2 = Errors.r2_predict(yi, self.labels)
        print(f'R^2 {method_type}: {r2}')

        ln = - res.fun
        res_logit_trivial = optimize.minimize(self.f_logit_trivial, self.a0,
                                              method='Nelder-Mead', options={'xtol': 1e-9, 'maxiter': 500})
        ln_trivial = - res_logit_trivial.fun

        pseudoR2 = Errors.pseudo_r2(ln, ln_trivial, self.example_number)
        print(f'pseudo R^2 {type}: {pseudoR2}')

        mcfaddenR2 = Errors.mcfadden_r2(ln, ln_trivial)
        print(f'McFadden R^2 {type}: {mcfaddenR2}')

    def get_test_result(self, x_test, y_test, method_type):
        if method_type == 'logit':
            res = self.res_logit
        else:
            res = self.res_probit
        theta = res.x

        example_number = x_test.shape[0]

        yi = []
        for i in range(example_number):
            val = self.g_logit(sum([theta[j] * x_test[i][j] for j in range(self.coefficient_number)]))
            if val > 0.5:
                yi.append(1)
            else:
                yi.append(0)

        r2 = Errors.r2_predict(yi, y_test)
        print(f'R^2 test {method_type}: {r2}')

    def predict(self, x_values, method_type):
        if method_type == 'logit':
            res = self.res_logit
        else:
            res = self.res_probit
        theta = res.x

        example_number = x_values.shape[0]

        yi = []
        for i in range(example_number):
            if method_type == 'logit':
                val = self.g_logit(sum([theta[j] * x_values[i][j] for j in range(self.coefficient_number)]))
            else:
                val = self.g_probit(sum([theta[j] * x_values[i][j] for j in range(self.coefficient_number)]))

            if val > 0.5:
                yi.append(1)
            else:
                yi.append(0)

        return yi

    def predict_proba(self, x_values, method_type):
        if method_type == 'logit':
            res = self.res_logit
        else:
            res = self.res_probit
        theta = res.x

        example_number = x_values.shape[0]

        yi = []
        for i in range(example_number):
            if method_type == 'logit':
                val = self.g_logit(sum([theta[j] * x_values[i][j] for j in range(self.coefficient_number)]))
            else:
                val = self.g_probit(sum([theta[j] * x_values[i][j] for j in range(self.coefficient_number)]))

            yi.append(val)

        return yi

    def process_probit(self, method):
        """
        Метод инициирует построение регрессионной модели с использованием Пробит-модели
        """
        res_probit = optimize.minimize(self.f_probit, self.a0, method=method,
                                       tol=1e-6, options={'disp': True, 'maxiter': 500})

        # print('Probit:')
        # print(res_probit.x)

        self.res_probit = res_probit

        return res_probit

    @staticmethod
    def g_logit(z):
        """
        Метод возвращает значение функции стандартного логистического распределения (Логит-модель)
        """
        return np.exp(z) / (1 + np.exp(z))

    @staticmethod
    def f1(t):
        """
        Вспомогательная подынтегральная функция для Пробит-модели
        :return double
        """
        return np.exp(-0.5 * (t ** 2))

    def g_probit(self, z):
        """
        Метод возвращает значение функции стандартного нормального распределенияя (Пробит-модель)
        """
        value = 1 / np.sqrt(2 * np.pi) * integrate.quad(self.f1, -np.inf, z)[0]
        return value

    def f_probit(self, theta):
        """
        Функция правдоподобия с использованием Пробит-модели
        :param theta: искомые коэффициенты регрессии
        :return  '-' значение функции (т.к. ищем максимум встроенными методами минимизации)
        """
        val = 0
        for i in range(self.example_number):
            yiL = 0
            for j in range(self.coefficient_number):
                yiL += theta[j] * self.data[i][j]
            if self.labels[i] == 1:
                val += self.labels[i] * math.log(self.g_probit(yiL))
            else:
                val += (1 - self.labels[i]) * math.log(1 - self.g_probit(yiL))
        return -val

    def f_logit(self, theta):
        """
        Функция правдоподобия с использованием Логит-модели
        :param theta: искомые коэффициенты регрессии
        :return  '-' значение функции (т.к. ищем максимум встроенными методами минимизации)
        """
        val = 0
        for i in range(self.example_number):
            yiL = 0
            for j in range(self.coefficient_number):
                yiL += theta[j] * self.data[i][j]
            if self.labels[i] == 1:
                val += self.labels[i] * math.log(self.g_logit(yiL))
            else:
                val += (1 - self.labels[i]) * math.log(1 - self.g_logit(yiL))
        return -val

    def f_probit_trivial(self, theta):
        """
        Функция правдоподобия для тривиальной модели с использованием Пробит-модели
        :param theta: искомые коэффициенты регрессии
        :return  '-' значение функции (т.к. ищем максимум встроенными методами минимизации)
        """
        val = 0
        for i in range(self.example_number):
            yiL = theta[0]

            if self.labels[i] == 1:
                val += self.labels[i] * math.log(self.g_probit(yiL))
            else:
                val += (1 - self.labels[i]) * math.log(1 - self.g_probit(yiL))
        return -val

    def f_logit_trivial(self, theta):
        """
        Функция правдоподобия для тривиальной модели с использованием Логит-модели
        :param theta: искомые коэффициенты регрессии
        :return  '-' значение функции (т.к. ищем максимум встроенными методами минимизации)
        """
        val = 0
        for i in range(self.example_number):
            yiL = theta[0]

            if self.labels[i] == 1:
                val += self.labels[i] * math.log(self.g_logit(yiL))
            else:
                val += (1 - self.labels[i]) * math.log(1 - self.g_logit(yiL))
        return -val
