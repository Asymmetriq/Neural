from computation import *
import itertools
import copy
from prettytable import PrettyTable


'''
Функция обучения
:param func_type: тип функции (пороговая или логист.)
:param return: таблица обучения и минимальный набор
'''
def learn_rbf(func_type):
    vlist, t1, f1 = get_Y_target(truth_table)

    # выбор центров
    if len(t1) <= 8:
        c_list = t1
    else:
        c_list = f1

    # пробегаем всевозможные комбинации
    for L in range(1, 17):
        for subset in itertools.combinations(truth_table, L):
            # заполняем веса
            vector_v = list()
            for i in range(0, len(c_list) + 1):
                vector_v.append(0)
            # целевой выход
            t_list, t2, f2 = get_Y_target(subset)
            error_list = list()
            epoch = 0
            epoch_list = list()

            pt = PrettyTable(['Epoch', 'Weights', 'Y', 'Error'])
            y_list = get_Y_real(vector_v, subset, func_type, c_list)
            epoch_list.append(epoch)
            error_sum = hamming_distance(t_list, y_list)
            error_list.append(error_sum)
            pt.add_row([epoch, copy.copy([float('{:.7f}'.format(x)) for x in vector_v]), y_list, error_sum])
            y_list = list()

            # пока ошибка не равна 0
            while error_sum != 0 and epoch < 50:
                epoch += 1

                for (vec, t) in zip(subset, t_list):
                    # расчёт фи
                    phi_list = list()
                    for c in c_list:
                        phi_list.append(get_phi(vec, c))
                    net = get_net(phi_list, vector_v) # считаем net

                    if func_type == 't':
                        y = get_function_out(net, 't')
                        y_list.append(y)
                    if func_type == 'l':
                        out = l_function(net)
                        y = get_function_out(out, 'l')
                        y_list.append(y)
                    # обновляем веса
                    delta = get_delta(t, y)
                    update_w(vector_v, delta, phi_list, net, func_type)

                epoch_list.append(epoch)
                y_list = get_Y_real(vector_v, subset, func_type, c_list)
                error_sum = hamming_distance(t_list, y_list)
                error_list.append(error_sum)
                pt.add_row([epoch, copy.copy([float('{:.10f}'.format(x)) for x in vector_v]), y_list, error_sum])
                y_list = list()
            # проверка весов на всех наборах
            y_list = get_Y_real(vector_v, truth_table, func_type, c_list)
            t_list, t3, f3 = get_Y_target(truth_table)

            error_sum = hamming_distance(t_list, y_list)
            if error_sum == 0:
                graph_plot(epoch_list, error_list)
                return pt, subset

    return pt


if __name__ == "__main__":
    table, comb = learn_rbf('l')
    print(table)
    print(comb)