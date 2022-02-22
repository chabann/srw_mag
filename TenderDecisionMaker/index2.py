import random
import matplotlib.pyplot as plt


def draw_difference(x_values, plan_values, color, label):
    plt.plot(x_values, plan_values, color=color, label=label, marker='_')


def sort_dict(dict_cost):
    return sorted(enumerate(dict_cost), key=lambda i: i[1])


def get_project_cost(start_cost):
    return int(random.randint(int(start_cost), int(start_cost) + 3) * 1000)


def get_cost(n):
    ar_costs = []

    for index in range(n):
        ar_costs.append(int(random.uniform(8, 15) * 1000))

    sorted_costs = sort_dict(ar_costs)
    return sorted_costs


def get_plan(n, items_good, start=0, proportion=1.0):
    percent = start
    percent_line = []
    n_count = n
    if n_count == 1:
        n_count = 2

    for point in range(n):
        percent_line.append(percent)

        if point < int(items_good):
            percent += ((1 - start) / (n_count - 1)) * proportion
        else:
            percent += ((1 - start) / (n_count - 1)) * random.uniform(0.55, 0.8) * proportion

        if percent > 1:
            percent = 1
    return percent_line


def get_next_cost(old_costs, index, number, fact_v):
    old_costs.pop(0)
    return [(old_costs[k][0], int(old_costs[k][1] * (1 - fact_v) * number / index)) for k in range(len(old_costs))]


def get_penalty(old_cost, n, i, f, b, i_k, n_k):
    sigma = random.uniform(1.2, 1.4)
    low_range = []
    old_cost.pop(0)
    for comp in old_cost:
        low_range.append(int(1 / n * (n - i) * sigma * comp[1] / (1 - f)))

    low_range.sort()
    delta = int(round((low_range[0] + low_range[1] - 2 * b * (1 - i_k / n_k)) / 2))
    if delta < 0:
        delta = 0
    return delta


def get_tender_data(n, m, d):
    sorted_costs = get_cost(m)
    first_pay = get_project_cost(sorted_costs[-1][1] / 1000)
    p = get_plan(n, n)
    p_fact = get_plan(n, d * n)
    return sorted_costs, first_pay, p, p_fact


def initialize_parameters():
    point_number = 20
    comp_number = 10
    k_break = 0.1
    return point_number, comp_number, k_break


def process_work(last_change, n, cur_cost, plan_, plan_fact, count_workers, costs_, last_stage_done, penalty, balance,
                 n0):
    cur_i = last_change
    for i in range(last_change, n0):
        cur_i = i
        if plan_[i] - plan_fact[i] > D:

            finishData[count_workers] = {
                'timeValues': list(range(last_change, n0)),
                'planValues': plan_fact[last_change:n0]
            }

            draw_difference(list(range(last_change, i + 1)), plan_fact[last_change:(i + 1)],
                            colors[count_workers + 1], 'Компания ' + str(count_workers))

            temp_balance = balance * (1 - (i - last_change) / n)
            penalty[count_workers] = get_penalty(cur_cost, n0, i, plan_fact[i], temp_balance, i - last_change, n)
            costs_[count_workers] = round(balance * (i - last_change) / n)  # - penalty[count_workers]  - без штрафов
            balance = round(balance * (1 - (i - last_change) / n) + penalty[count_workers])
            last_change = i
            last_stage_done = plan_fact[i]

            changesWorker[count_workers] = last_change
            changesWorkerDone[count_workers] = last_stage_done
            currentBudget[count_workers] = balance
            break
    return cur_i, last_stage_done, last_change, costs_, penalty, balance


def process_new_tender(plan_, plan_f, n_k, cur_i, d, count):
    current_cost = []
    new_cost = get_cost(M)
    for comp in new_cost:
        random_coef = random.uniform(1.5, 1.8) * (N - cur_i) / N
        current_cost.append((comp[0], comp[1] * random_coef))

    new_plan = get_plan(n_k, n_k, plan_f[cur_i])
    n_plan = plan_[0:cur_i] + [new_plan[_] for _ in range(n_k)]

    if count < 4:
        new_plan_f = get_plan(n_k + 1, d * (n_k + 1), plan_f[cur_i])
    else:
        new_plan_f = get_plan(n_k, n_k, plan_f[cur_i])
    n_plan_f = plan_f[0:cur_i] + [new_plan_f[_] for _ in range(n_k)]

    if 1 - n_plan_f[-1] < 0.05:
        n_plan_f[-1] = 1
    return current_cost, n_plan, n_plan_f


if __name__ == "__main__":
    N, M, D = initialize_parameters()  # начальные параметры
    startCost, budget, plan, planFact = get_tender_data(N, M, D)  # параметры текущего тендера
    N_k = N
    changesWorker = {}
    changesWorkerDone = {}
    currentBudget = {}
    countWorkers = 0
    currentStage = 0
    lastItemChange = 0
    lastStageDone = 0
    finishData = {}
    penalties = {}
    plan_start = plan.copy()
    colors = ['b', 'g', 'y', 'c', 'red', 'limegreen', 'firebrick', 'coral', 'mediumblue', 'aqua', 'goldenrod',
              'chocolate', 'magenta', 'teal', 'fuchsia', 'lavender', 'olive', 'lightgreen', 'tan', 'violet',
              'salmon', 'indigo']
    costs = {}
    actualPlans = {
        0: [0, 0.05, 0.08, 0.111, 0.131, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
        1: [0, 0.05, 0.08, 0.111, 0.131, 0.15, 0.18, 0.22, 0.23, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24],
        2: [0, 0.05, 0.08, 0.111, 0.131, 0.15, 0.18, 0.22, 0.23, 0.30, 0.35, 0.39, 0.40, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
        3: [0, 0.05, 0.08, 0.111, 0.131, 0.15, 0.18, 0.22, 0.24, 0.30, 0.35, 0.39, 0.40, 0.42, 0.51, 0.60, 0.70, 0.79, 0.91, 1.0]
    }
    draw_difference(list(range(N)), plan[0:N], colors[0], 'Плановые показатели')
    currentBudget[0] = budget

    while currentStage < (N - 1):
        planFact = actualPlans[countWorkers]
        countWorkers += 1
        currentStage, lastStageDone, lastItemChange, costs, penalties, budget = process_work(lastItemChange, N_k,
                                                                                             startCost, plan,
                                                                                             planFact, countWorkers,
                                                                                             costs,
                                                                                             lastStageDone,
                                                                                             penalties, budget, N)
        N_k = N - lastItemChange
        startCost, plan, planFact = process_new_tender(plan, planFact, N_k, lastItemChange, D, countWorkers + 1)

    changesWorkerDone[countWorkers] = planFact[-1]
    costs[countWorkers] = budget * (N - lastItemChange) / N_k
    penalties[countWorkers] = 0

    budget = budget * (1 - (N - lastItemChange) / N_k)
    currentBudget[countWorkers] = budget

    draw_difference(list(range(lastItemChange, N)), planFact[lastItemChange:N],
                    colors[countWorkers + 1], 'Компания ' + str(countWorkers))

    plt.title('План выполнения проекта')
    plt.legend()
    plt.grid()
    plt.show()

    print('')
    print('Плата подрядчикам: ', costs)
    print('')
    print('Штрафы подрядчикам: ', penalties)
    print('')
    print('Точки замены: ', changesWorker)
    print('')
    print('Выполненная работа: ', changesWorkerDone)
    print('')
    print('Остатки бюджета: ', currentBudget)
    print('')
    print('Изначальный план: ', plan_start)
