from math import sin, tan
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib as mpl


'''
Исходная функция 
:param t: параметр t
:param return: значение функции
'''
def function(t):
    # return t**4 + 2*t**3 + t
    return (sin(t)) ** 2


'''
Функция вычисления эпсилон 
:param dlist: набор дельт
:param return: значение эпсилон
'''
def get_epsilon(dlist):
    epsilon = 0
    for delta in dlist:
        epsilon += delta**2
    return sqrt(epsilon)


'''
Функция вычисления коррекции веса 
:param delta: дельта
:param return: значение коррекции
'''
def get_dw(delta, norm, x):
    return norm * delta * x


'''
Функция построения графика
:param x_array: значения x
:param y_array1: первый набор значений y
:param y_array2: второй набор значений y (для типа 1)
:param start_x: начальная точка x
:param start_y: начальная точка y
:param label_x: название гор. оси
:param label_y: название верт. оси
:param plot_type: тип графика
'''
def graph_plot(x_array, y_array1, y_array2, start_x, start_y, label_x, label_y, plot_type):
    style.use('seaborn')
    style.use('ggplot')
    plt.grid(True)
    if plot_type == 1:
        plt.plot(x_array, y_array2)
    plt.plot(x_array[start_x:], y_array1[start_y:], 'o')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    mpl.style.use('bmh')
    plt.show()

