import numpy

'''
Вспомогательная функция печати символа
:param s: единичная клетка матрицы
:param newline: флаг переноса на новую строку
'''
def visual_print(s, newline):
    if s == 1:
        if newline == 'n':
            print("%3s" % s, end='')
        else:
            print("%3s" % s)
    else:
        if newline == 'n':
            print("%3s" % ' ', end='')
        else:
            print("%3s" % ' ')


'''
Функция печати паттерна
:param pattern: паттерн
'''
def print_pattern(pattern):
    for i in range(5):
        index = i
        for j in range(2):
            visual_print(pattern[index], 'n')
            index += 5
        visual_print(pattern[index], 'y')


'''
Функция получения матрицы весов 
:param patterns: паттерны
:param return: найденный вектор
'''
def get_matrix(patterns):
    matrix = numpy.zeros((len(patterns[0]), len(patterns[0])))
    for i in range(len(patterns[0])):
        for j in range(len(patterns[0])):
            matrix[i][j] = get_weight(patterns, i, j)
    return matrix


'''
Функция подсчитывает f(net)
:param y_prev: значение с предыдущей эпохи
:param net: net
:param return: значение f(net)
'''
def function(y_prev, net):
    if net > 0:
        return 1
    elif net == 0:
        return y_prev
    else:
        return -1


'''
Функция полуения веса
:param patterns: векторизированные паттерны
:param i: итератор i
:param j: итератор j
:param return: вес
'''
def get_weight(patterns, i, j):
    result = 0
    if i != j:
        for s in patterns:
            result += s[i] * s[j]
    return result
