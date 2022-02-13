import random
import matplotlib.pyplot as plt


def draw_difference(x_values, plan_values, color, label):
    plt.plot(x_values, plan_values, color + '-', label=label)


def sort_dict(dict_cost):
    return sorted(enumerate(dict_cost), key=lambda i: i[1])


def get_project_cost(start_cost):
    return int(random.randint(int(start_cost), 40) * 10000)


def get_cost(n):
    ar_costs = []

    for index in range(n):
        ar_costs.append(int(random.uniform(8, 15) * 1000))

    sorted_costs = sort_dict(ar_costs)
    return sorted_costs


def get_plan(n, items_good, add=0, proportion=1.0):
    percent = add
    percent_line = []
    n_count = n
    if n_count == 1:
        n_count = 2

    for point in range(n):
        percent_line.append(percent)

        if point < int(items_good):
            percent += (1 / (n_count - 1)) * proportion
        else:
            percent += (1 / (n_count - 1)) * random.uniform(0.55, 1) * proportion
            """if point < int(n_count * (items_good / n_count + (1 - items_good / n_count) / 2)):
                percent += (1 / (n_count - 1)) * random.uniform(0.7, 1) * proportion
            else:
                percent += (1 / (n_count - 1)) * random.uniform(0.4, 0.7) * proportion"""

        if percent > 1:
            percent = 1
    return percent_line


def get_next_cost(old_costs, index, number, fact_v):
    old_costs.pop(0)
    return [(old_costs[k][0], int(old_costs[k][1] * (1 - fact_v) * number / index)) for k in range(len(old_costs))]


def get_penalty(old_cost, n, i, d, b):
    sigma = random.uniform(1.2, 1.4)
    low_range = []
    old_cost.pop(0)
    print('old cost', old_cost)
    for comp in old_cost:
        # print((n-i) * comp[1] / (n - i * (1 - d)))
        low_range.append(int((n-i) * sigma * comp[1] / (n - i * (1 - d))))

    print('low_range', low_range)
    low_range.sort()
    print('low_range_sort', low_range)
    print('low bound', low_range[0] - b * (1 - i/n))
    print('up bound', low_range[1] - b * (1 - i / n))
    delta = int(round((low_range[0] + low_range[1] - 2 * b * (1 - i/n)) / 2))
    print('delta', delta)
    return delta


def initialize_parameters():
    comp_number = 10
    point_number = 20
    k_break = 0.1
    sorted_costs = get_cost(comp_number)
    first_pay = get_project_cost(sorted_costs[0][1] / 10000)
    p = get_plan(point_number, point_number)
    p_fact = get_plan(point_number, k_break * point_number)
    return point_number, comp_number, k_break, sorted_costs, first_pay, p, p_fact


if __name__ == "__main__":
    N, M, k, startCost, firstPay, plan, planFact = initialize_parameters()
    countWorkers = 1
    lastItemChange = 0
    lastStageDone = 0
    finishData = {}
    penalties = {}
    colors = ['b', 'g', 'y', 'c', 'r']
    costs = {}

    firstPay = 14000

    print('')
    print('Изначальная стоимость проекта', firstPay)
    print('')
    print('Ранжированный список подрядчиков', startCost)
    print('')

    currentCost = startCost.copy()
    planStart = plan.copy()
    for i in range(N):
        if plan[i] - planFact[i] > 0.1:
            print('Контрольная точка ', i)
            print('Отставание от плана: ', round((plan[i] - planFact[i]) * 100, 1), '%')
            print('Меняем подрядчика')
            print('')

            finishData[countWorkers] = {
                'timeValues': list(range(lastItemChange, i+1)),
                'planValues': planFact[lastItemChange:i+1]
            }

            costs[countWorkers] = int(firstPay * i / N)
            color_count = countWorkers % 5
            draw_difference(list(range(lastItemChange, N)), planFact[lastItemChange:N],
                            colors[color_count], 'Компания ' + str(countWorkers))

            penalties[countWorkers] = get_penalty(currentCost, N, i, planFact[i], firstPay)

            balance = firstPay - costs[countWorkers] + penalties[countWorkers]
            print('balance', balance)
            # currentCost = get_next_cost(startCost, N - i, N, planFact[i])
            currentCost = []
            newCost = get_cost(M - 1)
            for comp in newCost:
                randomCoef = random.uniform(1.5, 1.8) * (N - i) / N
                currentCost.append((comp[0], comp[1] * randomCoef))
                # comp[1] *= randomCoef

            # print('Коэффициент надбавки стоимости ', round((1 - planFact[i]) * N / (N - i), 2))
            print('')
            print('Новый список подрядчиков', currentCost)

            newPlan = get_plan(N - i, N - i, planFact[i], (1 - planFact[i]))
            planFact = planFact[0:i] + [newPlan[_] for _ in range(N - i)]
            plan = plan[0:i] + [newPlan[_] for _ in range(N - i)]

            # balance = int(firstPay - costs[countWorkers] - currentCost[0][1] * (1 - planFact[i]))
            # if balance < 0:
            #     penalties[countWorkers] = abs(balance)
            #else:
            #     penalties[countWorkers] = 0

            countWorkers += 1
            lastItemChange = i
            lastStageDone = planFact[i]

    if 1 - planFact[-1] < 0.05:
        planFact[-1] = 1

    planFact[-1] = 1
    if lastItemChange > 0:
        finishData[countWorkers] = {
            'timeValues': list(range(lastItemChange, N)),
            'planValues': planFact[lastItemChange:N]
        }
        # costs[countWorkers] = int(currentCost[0][1] * (1 - lastStageDone))
        costs[countWorkers] = balance
        color_count = countWorkers % 5
        draw_difference(list(range(lastItemChange, N)), planFact[lastItemChange:N],
                        colors[color_count], 'Компания ' + str(countWorkers))

    color_count = countWorkers + 1 % 5
    draw_difference(list(range(N)), planStart[0:N],
                    colors[color_count], 'Плановые показатели')

    plt.title('План выполнения проекта')
    plt.legend()
    plt.grid()
    plt.show()

    print('')
    print('Плата подрядчикам: ', costs)
    print('')
    print('Штрафы подрядчикам: ', penalties)
    # print('finishData', finishData)
