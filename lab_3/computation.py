from math import exp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.style as style

truth_table = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
               [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
               [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
               [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]]
nu = 0.3    # норма обучения


'''
Функция подсчитывает net
:param vector_x: булеый набор
:param vector_w: набор весов
:param return: значение net
'''
def get_net(vector_x, vector_w):
    net = 0
    for i in range(1, len(vector_w)):
        net += vector_x[i-1] * vector_w[i]
    net += vector_w[0]
    return net


'''
Функция пересчитывает значения весов
:param vector_w: набор весов
:param delta: дельта
:param vec: булевый набор
:param net: net
:param func_type: тип функции (пороговая или логист.)
:param return: новый набор весов
'''
def update_w(vector_w, delta, vec, net, func_type):
    for i in range(1, len(vector_w)):
        vector_w[i] += get_dw(delta, nu, vec[i - 1], net, func_type)
    vector_w[0] += get_dw(delta, nu, 1, net, func_type)
    return vector_w


'''
Функция подсчитывает f(net)
:param net: net
:param return: значение f(net)
'''
def l_function(net):
    return 1 / (1 + exp(-net))


'''
Функция подсчитывает y(net) / y(out)
:param net_out:  значение net / out
:param func_type: тип функции (пороговая или логист.)
:param return: значение f(net) / y(out)
'''
def get_function_out(net_out, func_type):
    if func_type == 't':
        return 1 if net_out >= 0 else 0
    else:
        return 1 if net_out >= 0.5 else 0


'''
Функция подсчитывает набор реальных выходов
:param vector_w: набор весов
:param table: все булевы наборы
:param func_type: тип функции (пороговая или логист.)
:param return: набор реальных выходов
'''
def get_Y_real(vector_v, table, func_type, c_list):
    y_list = list()
    for vec in table:

        phi_list = list()
        for c in c_list:
            phi_list.append(get_phi(vec, c))
        net = get_net(phi_list, vector_v)

        if func_type == 't':
            y = get_function_out(net, 't')
            y_list.append(y)
        if func_type == 'l':
            out = l_function(net)
            y = get_function_out(out, 'l')
            y_list.append(y)
    return y_list


'''
Функция подсчитывает набор целевых выходов
:param table: все булевы наборы
:param return: набор целевых выходов
'''
def get_Y_target(table):
    t_list = list()
    truth_list = list()
    false_list = list()
    for vec in table:
        t_list.append(int(bool_function(vec)))
        if int(bool_function(vec)) == 1:
            truth_list.append(vec)
        else:
            false_list.append(vec)
    return t_list, truth_list, false_list


'''
Функция подсчитывает дельта
:param t: целевой выход
:param y: реальный выход
:param return: значение delta
'''
def get_delta(t, y):
    return t - y


def get_phi(vector_x, vector_c):
    sm = 0
    for i in range(0, 4):
        sm += (vector_x[i] - vector_c[i])**2
    return exp(-sm)


'''
Функция подсчитывает коррекцию веса
:param delta: дельта
:param n: норма обучения
:param x: компонента обучающего вектора
:param net: net
:param func_type: тип функции (пороговая или логист.)
:param return: значение коррекции веса
'''
def get_dw(delta, n, x, net, func_type):
    if func_type == 'l':
        return n * delta * l_function(net) * (1 - l_function(net)) * x
    else:
        return n * delta * x


'''
Функция подсчитывает квадратичную ошибку
:param vec1: целевой набор выходов
:param vec2: реальный набор выходов
:param return: значение квадратичной ошибки
'''
def hamming_distance(vec1, vec2):
    return sum(ch1 != ch2 for ch1, ch2 in zip(vec1, vec2))


'''
Функция подсчитывает значение булевой функции
:param vector: булевый набор
:param return: значение булевой функции
'''
def bool_function(vector):
    return (vector[0] or vector[1] or vector[3]) and vector[2]
    # return not ((vector[0]) and (vector[1])) and vector[2] and vector[3]


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
