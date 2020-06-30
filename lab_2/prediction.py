from computation import *
import numpy as np


'''
Функция обучения 
:param points_number: число точек интервала
:param window_size: размер окна
:param norm: норма обучения
:param vector_w: набор весов
:param epoch_max: максимальное число эпох
:param return: набор весов, последняя погрешность
'''
def sliding_window_learn(points_number, window_size, norm, vector_w, epoch_max):
    points = np.linspace(0, 4, 40)
    x = int(window_size)

    values = list()
    for pt in points[0:20]:
        values.append(function(pt))

    epoch = 0

    while epoch < epoch_max:

        s = 0
        f = x

        while f < points_number:

            # подсчёт net
            net = 0
            for (v, w) in zip(values[s:f], vector_w[1:]):
                net += v * w
            net += vector_w[0]

            # Вычисление delta
            delta = values[f] - net

            # Пересчёт весовых коэффициентов
            for (j, v) in zip(range(1, len(vector_w)), range(s, f)):
                vector_w[j] += get_dw(delta, norm, values[v])
            vector_w[0] += get_dw(delta, norm, 1)

            s += 1
            f += 1

        epoch += 1

    return vector_w


'''
Функция прогнозирования графика 
:param points_number: число точек интервала
:param window_size: размер окна
:param vector_w: набор весов
:param return: спрогнозированные значения функции
'''
def predictive_plot(points_number, window_size, vector_w):
    points = pts = np.linspace(0, 4, 40)
    x = int(window_size)

    y = list()
    for pt in points:
        y.append(function(pt))

    values = list()
    new_points = points[(20 - x):20]
    for pt in new_points:
        values.append(function(pt))

    sec_values = list()
    sec_points = points[20:]
    for pt in sec_points:
        sec_values.append(function(pt))

    s = 0
    f = x

    delta_list = list()

    while f < points_number + x:

        net = 0
        for (v, w) in zip(values[s:], vector_w[1:]):
            net += v * w
        net += vector_w[0]
        values.append(net)

        delta = sec_values[s] - net
        delta_list.append(delta)

        s += 1
        f += 1

    epsilon = get_epsilon(delta_list)

    graph_plot(points, values, y, 20, x, 'x', 'x(t)', 1)

    return values, epsilon


'''
Функция тестирования для отчёта 
:param N: число точек интервала
:param ws: размер окна
:param W: набор весов
:param M: максимальное число эпох
'''
def test(N, ws, W, M):
    norm_list = list()
    e_list = list()
    # for epoch in range(100, 1500, 100):
    # for norm in np.arange(0.1, 0.7, 0.05):
    for w_size in range(4, 20):
        W = [0,0,0,0,0]
        new_W = sliding_window_learn(N, w_size, 0.7, W, 500)
        vc, e = predictive_plot(N, w_size, new_W)
        norm_list.append(w_size)
        e_list.append(e)
    style.use('seaborn')
    style.use('ggplot')
    plt.plot(norm_list, e_list)
    plt.xlabel('window size, p')
    plt.ylabel('epsilon')

    plt.grid(True)
    plt.show()



if __name__ == "__main__":

    N = int(input('Enter N:'))
    M = int(input('Enter M:'))
    p = int(input('Enter sliding window size:'))
    nu = float(input('Enter nu:'))

    W = list()
    for i in range(0, p + 1):
        W.append(0)

    vec = sliding_window_learn(N, p, nu, W, M)
    temp, eps = predictive_plot(N, p, vec)

    W = list()
    for i in range(0, p + 1):
        W.append(0)

    # test(N, p, W, M)
    print(vec)
    print(eps)
