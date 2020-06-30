from computation import *
from prettytable import PrettyTable

'''
Функция обучения
:param N: параметр N
:param J: параметр J
:param M: параметр M
:param vector_x: входной вектор
:param vector_t: целевой вектор
:param eps: погрешность
'''
def learn_mnn(N, J, M, vector_x, vector_t, eps):

    # генерируем начальные веса скрытых нейронов
    hnw = list()    # hidden neuron weights
    for i in range(0, J):
        hnw.append([0.0] * (N + 1))
    # генерируем начальные веса внешних нейронов
    enw = list()    # external neuron weights
    for i in range(0, M):
        enw.append([0.0] * (J + 1))

    epoch_list = list()
    error_list = list()
    epoch = 0
    epoch_list.append(epoch)

    y = [0.0] * (M)
    error = count_error(y, vector_t)
    error_list.append(error)

    pt = PrettyTable(['Epoch','Weights (hidden)', 'Weights (external)', 'Y', 'E'])
    pt.add_row([epoch,
                [[float('{:.5}'.format(hw)) for hw in hnw[i]] for i in range(len(hnw))],
                [[float('{:.5}'.format(ew)) for ew in enw[j]] for j in range(len(enw))],
                [float('{:.5}'.format(y)) for y in y], round(error, 5)])

    while error > eps:

        # ПЕРВЫЙ ЭТАП (1)
        # получаем значения нейронов скрытого и внешнего слоёв
        h_nets, h_outs, e_nets, e_outs \
            = get_neuron_values(hnw, enw, vector_x)

        # ВТОРОЙ ЭТАП (2)
        # внешние дельта
        e_deltas = get_external_deltas(e_nets, e_outs, vector_t)
        # скрытые дельта
        h_deltas = get_hidden_deltas(J, enw, e_deltas, h_nets)

        # ТРЕТИЙ ЭТАП (3)
        # пересчёт скрытых весов
        hnw = update_hidden_weights(hnw, J, N, h_deltas, vector_x)
        # пересчёт внешних весов
        enw = update_external_weights(enw, M, J, e_deltas, h_outs)

        # пересчитыаем Y
        h_nets, h_outs, e_nets, \
        y = get_neuron_values(hnw, enw, vector_x)

        error = count_error(y, vector_t)
        epoch += 1
        epoch_list.append(epoch)
        error_list.append(error)

        pt.add_row([epoch,
                       [[float('{:.5}'.format(hw)) for hw in hnw[i]] for i in range(len(hnw))],
                       [[float('{:.5}'.format(ew)) for ew in enw[j]] for j in range(len(enw))],
                       [float('{:.5}'.format(y)) for y in y], round(error, 5)])

    graph_plot(epoch_list, error_list)
    return pt


if __name__ == "__main__":
    X = [1, -2]
    T = [0.2, 0.1, 0.3]
    test = learn_mnn(1, 1, 3, X, T, 0.0001)
    print(test)
