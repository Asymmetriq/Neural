from computation import *
import itertools
import copy
from prettytable import PrettyTable


'''
Функция обучения на всём числе наборов
:param vector_w: набор весов
:param func_type: тип функции (пороговая или логист.)
:param return: таблица обучения
'''
def learn(vector_w, func_type):
    y_list = get_Y_real(vector_w, truth_table, func_type)
    t_list = get_Y_target(truth_table)

    epoch_list = list()
    error_list = list()
    error_sum = hamming_distance(y_list, t_list)
    epoch = 0

    pt = PrettyTable(['K', 'W', 'Y', 'E'])
    pt.add_row([epoch, copy.copy(vector_w), y_list, error_sum])

    # пока ошибка не равна 0
    while error_sum != 0:
        y_list = list()
        epoch += 1

        for (vec, t) in zip(truth_table, t_list):

            net = get_net(vec, vector_w)
            if func_type == 't':
                y = get_function_out(net, 't')
                y_list.append(y)
            if func_type == 'l':
                out = l_function(net)
                y = get_function_out(out, 'l')
                y_list.append(y)

            delta = get_delta(t, y)
            update_w(vector_w, delta, nu, vec, net, func_type)

        epoch_list.append(epoch)

        y_list = get_Y_real(vector_w, truth_table, func_type)
        error_sum = hamming_distance(t_list, y_list)
        error_list.append(error_sum)
        pt.add_row([epoch, copy.copy([float('{:.2f}'.format(x)) for x in vector_w]), y_list, error_sum])

    graph_plot(epoch_list, error_list)
    return pt


'''
Функция обучения на минималном числе наборов
:param vector_w: набор весов
:param func_type: тип функции (пороговая или логист.)
:param return: таблица обучения
'''
def learn_min(vector_w, func_type):
    error_sum = 1
    for L in range(1, 17):
        for subset in itertools.combinations(truth_table, L):

            vector_w = [0, 0, 0, 0, 0]
            y_list = get_Y_real(vector_w, subset, func_type)
            t_list = get_Y_target(subset)
            error_list = list()
            epoch = 0
            epoch_list = list()

            pt = PrettyTable(['K', 'W', 'Y', 'E'])

            # пока ошибка не равна 0
            while error_sum != 0 and epoch < 50:
                epoch += 1

                for (vec, t) in zip(subset, t_list):

                    net = get_net(vec, vector_w)
                    if func_type == 't':
                        y = get_function_out(net, 't')
                        y_list.append(y)
                    if func_type == 'l':
                        out = l_function(net)
                        y = get_function_out(out, 'l')
                        y_list.append(y)

                    delta = get_delta(t, y)
                    update_w(vector_w, delta, nu, vec, net, func_type)

                epoch_list.append(epoch)
                y_list = get_Y_real(vector_w, subset, func_type)
                error_sum = hamming_distance(t_list, y_list)
                error_list.append(error_sum)
                pt.add_row([epoch, copy.copy([float('{:.4f}'.format(x)) for x in vector_w]), y_list, error_sum])
                y_list = list()

            y_list = get_Y_real(vector_w, truth_table, func_type)
            t_list = get_Y_target(truth_table)
            error_sum = hamming_distance(t_list, y_list)
            if error_sum == 0:
                graph_plot(epoch_list, error_list)
                return pt, subset

    return pt


if __name__ == "__main__":
    W = [0, 0, 0, 0, 0]
    test, combs = learn_min(W, 'l')
    # test = learn(W, 'l')
    print(test)
    print(combs)