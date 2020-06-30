from math import exp, sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.style as style

nu = 1

'''
Функция подсчитывает net
:param vector_x: входной вектор
:param vector_w: вектор весов
:param return: значение net
'''
def get_net(vector_x, vector_w):
    net = 0
    for i in range(1, len(vector_w)):
        net += vector_x[i] * vector_w[i]
    net += vector_w[0]
    return net


'''
Функция подсчитывает f(net)
:param net: net
:param return: значение f(net)
'''
def function(net):
    return (1 - exp(-net)) / (1 + exp(-net))


'''
Функция подсчитывает производную функции
:param net: net
:param return: значение производной
'''
def derivative(net):
    return 0.5 * (1 - function(net)**2)


'''
Функция рассчитывает дельты внешних нейронов
:param e_nets: net'ы внешних нейронов
:param e_outs: выходы внешних нейронов
:param vector_t: целевой вектор
:param return: набор дельт
'''
def get_external_deltas(e_nets, e_outs, vector_t):
    # внешние дельта
    external_deltas = list()
    for (net, out, t) in zip(e_nets, e_outs, vector_t):
        external_deltas.append(derivative(net) * (t - out))  # считаем внешние delta
    return external_deltas


'''
Функция рассчитывает дельты скрытых нейронов
:param J: параметр J
:param enw: веса внешних нейрнонов
:param e_deltas: дельты внешних нейронов
:param h_nets: net'ы скрытых нейронов
:param return: набор дельт
'''
def get_hidden_deltas(J, enw, e_deltas, h_nets):
    # вычисление подсумм
    sums = list()
    for j in range(0, J):
        temp = 0
        for (weight_m, delta) in zip(enw, e_deltas):
            temp += weight_m[j] * delta
        sums.append(temp)

    # скрытые дельта
    hidden_deltas = list()
    for (net, s) in zip(h_nets, sums):
        hidden_deltas.append(derivative(net) * s)  # считаем скрытые delta
    return hidden_deltas


'''
Функция расчёта net и выходов скрытых и внешних нейронов
:param hnw: веса скрытых нейрнонов
:param enw: веса внешних нейрнонов
:param vector_x: входной вектор
:param return: net'ы и выходы
'''
def get_neuron_values(hnw, enw, vector_x):

    # скрытые выходы
    hidden_nets = list()  # net'ы скрытого слоя
    hidden_outs = list()  # выходы скрытого слоя
    hidden_outs.append(1)  # добавляем x смещения
    for hidden_weights in hnw:  # для каждого набора весов
        net = get_net(vector_x, hidden_weights)  # считаем net
        hidden_nets.append(net)
        hidden_outs.append(function(net))  # считаем выход скрытого слоя

    # внешние выходы
    external_nets = list()  # net'ы внешнего слоя
    external_outs = list()  # выходы внешнего слоя
    for external_weights in enw:  # для каждого набора весов
        net = get_net(hidden_outs, external_weights)  # считаем net
        external_nets.append(net)
        external_outs.append(function(net))  # считаем выход внешнего слоя

    return hidden_nets, hidden_outs, external_nets, external_outs


'''
Функция обновления весов скрытых нейронов
:param hnw: веса скрытых нейрнонов
:param J: параметр J
:param N: параметр N
:param h_deltas: дельты скрытых нейронов
:param vector_x: входной вектор
:param return: обновлённые веса
'''
def update_hidden_weights(hnw, J, N, h_deltas, vector_x):
    # пересчёт скрытых весов
    for j in range(0, J):
        for i in range(0, N):
            hnw[j][i] += nu * vector_x[i] * h_deltas[j]
    return hnw


'''
Функция обновления весов внешних нейронов
:param enw: веса внешних нейрнонов
:param M: параметр M
:param J: параметр J
:param e_deltas: дельты внешних нейронов
:param h_outs: выходы скрытых нейронов
:param return: обновлённые веса
'''
def update_external_weights(enw, M, J, e_deltas, h_outs):
    # пересчёт внешних весов
    for m in range(0, M):
        for j in range(0, J):
            enw[m][j] += nu * h_outs[j] * e_deltas[m]
    return enw


'''
Функция подсчёта квадратичной ошибки
:param y: вектор y
:param vector_t: целевой вектор
:param return: квадратичная ошибка
'''
def count_error(y, vector_t):
    error = 0
    for (out, t) in zip(y, vector_t):
        error += (t - out) ** 2
    error = sqrt(error)
    return error


'''
Функция строит график E(k)
:param epoch_list: список эпох
:param error_list: список ошибок
'''
def graph_plot(epoch_list, error_list):
    style.use('seaborn')
    style.use('ggplot')
    plt.grid(True)
    plt.plot(epoch_list, error_list)
    plt.xlabel('Era k')
    plt.ylabel('Error E')
    plt.scatter(epoch_list, error_list)
    mpl.style.use('bmh')
    plt.show()
