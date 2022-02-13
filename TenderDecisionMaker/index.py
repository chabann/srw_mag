import numpy as np
import random
from scipy import interpolate
import matplotlib.pyplot as plt


def set_initial_parameters():
    point_count = 20
    part_good = 0.2
    cost1 = 10000
    penalty = 1.3
    loyalty_level = 0.3
    percent = 0
    percent_f = 0

    percent_plan = []
    percent_fact = []

    for point in range(point_count):
        percent_plan.append(percent)
        percent_fact.append(percent_f)

        if point < int(point_count * part_good):
            percent_f += (1 / (point_count - 1))
        else:
            if point < int(point_count * (part_good + (1 - part_good) / 2)):
                percent_f += (1 / (point_count - 1)) * random.uniform(0.6, 0.8)
            else:
                percent_f += (1 / (point_count - 1)) * random.uniform(0.4, 0.6)

        percent += (1 / (point_count - 1))

        if percent_f > 1:
            percent_f = 1

    return cost1, penalty, percent_fact, percent_plan, point_count, loyalty_level


def get_extrapolate_points(actual, point, is_correct):
    x_values = list(range(point + 1))
    expected = [actual[_] for _ in x_values]

    predict_func = interpolate.interp1d(x_values, expected, fill_value="extrapolate")

    for future_point in range(point + 1, N):
        if isinstance(predict_func(future_point), np.ndarray):
            next_point = predict_func(future_point).item(0)
        else:
            next_point = predict_func(future_point)

        diff = next_point - expected[future_point - 1]
        diff_plan = f_plan[future_point] - f_plan[future_point - 1]

        if is_correct:
            overfulfilment = random.uniform(1, 1.3)
            probability = random.random()
            prop_higher = loyalty / 3

            if probability < loyalty:
                if probability < prop_higher:
                    diff = diff_plan * overfulfilment
                else:
                    diff = diff_plan

        if expected[future_point - 1] + diff < expected[future_point - 1]:
            diff = diff_plan * random.uniform(0.85, 1)

        if expected[future_point - 1] + diff > f_plan[future_point]:
            diff = f_plan[future_point] - expected[future_point - 1]

        expected.append(expected[future_point - 1] + diff)

    return expected


def get_cost_second_comp(point):
    cost = c1 * (1.2 + point / N)
    return cost


def get_own_penalty(delay):
    cost = c1 * delay * 3
    return cost


def get_expenses_if_break(point):
    c2 = get_cost_second_comp(point)
    of1 = c1 * f_plan[point] - (f_plan[point] - f_fact[point]) * p + c2 * (1 - f_fact[point])
    return of1


def get_expenses_if_not_break():
    s = get_own_penalty(f_plan[N-1] - f_extra[N-1])
    of2 = c1 - (f_plan[N-1] - f_extra[N-1]) * p + s
    return of2


def draw_difference(point):
    plt.figure()
    plt.plot(xValues, f_extra, 'b--', label='predict')
    plt.plot(xValues, f_plan, 'g--', label='plan')
    plt.plot(xValues, f_fact, 'r--', label='fact')
    plt.title('extrapolate for point ' + str(point) + ' of ' + str(N))
    plt.legend()
    plt.grid()


if __name__ == "__main__":
    false = 0
    true = 1

    c1, p, f_fact, f_plan, N, loyalty = set_initial_parameters()
    print('cost1: ', c1)
    print('penalty coefficient for company 1: ', p)
    print('f plan: ', f_plan)
    print('f fact: ', f_fact)
    print('loyalty level: ', loyalty)

    xValues = list(range(N))

    print(' ')
    print('Calculate without loyalty:')
    print(' ')
    for i in range(N):
        if f_plan[i] - f_fact[i] > 0:
            f_extra = get_extrapolate_points(f_fact, i, false)
            # print('final extra: ', i, ' : ', f_extra[-1])

            if get_expenses_if_break(i) < get_expenses_if_not_break():
                print('expenses if break at step: ', i, ' = ', get_expenses_if_break(i))
                print('expenses if continue at step: ', i, ' = ', get_expenses_if_not_break())
                print('Break the contract at step: ', i)
                draw_difference(i)
                break
            else:
                print('expenses if break at step: ', i, ' : ', get_expenses_if_break(i))

                if i == (N - 1):
                    draw_difference(i)

    print(' ')
    print('Calculate with loyalty:')
    print(' ')
    for i in range(N):
        if f_plan[i] - f_fact[i] > 0:
            f_extra = get_extrapolate_points(f_fact, i, true)
            # print('final extra: ', i, ' : ', f_extra[-1])

            if get_expenses_if_break(i) < get_expenses_if_not_break():
                # print('expenses if break: ', c1 * f_plan[i] - (f_plan[i] - f_fact[i]) * p,
                # ' + ', get_cost_second_comp(i) * (1 - f_fact[i]))

                print('expenses if break at step: ', i, ' = ', get_expenses_if_break(i))
                print('expenses if continue at step: ', i, ' = ', get_expenses_if_not_break())
                print('Break the contract at step: ', i)
                draw_difference(i)
                break
            else:
                print('expenses if break at step: ', i, ' : ', get_expenses_if_break(i))
                print('expenses if continue at step: ', i, ' : ', get_expenses_if_not_break())

                if i == N - 1:
                    draw_difference(i)
    plt.show()
