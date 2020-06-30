from computation import *

vectorized_pattern_1 = [-1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1]
vectorized_pattern_2 = [1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1]
vectorized_pattern_3 = [1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1]

'''
Функция восстановления эталона
:param vector_x: входной вектор
:param vector_target: эталон
:param matrix_w: матрица весов обратных связей
:param return: найденный вектор
'''
def recover_RNN(vector_x, vector_target, matrix_w):
    vector_y = vector_x
    vector_prev = list()

    temp = list()
    epoch = 0
    while vector_y != temp:
        temp = vector_target
        if epoch == 0:
            vector_prev = vector_y
        epoch += 1

        y_current = list()
        for k in range(len(vector_x)):
            sum_1 = 0
            for j in range(k - 1):
                sum_1 += matrix_w[j][k] * vector_y[j]
            sum_2 = 0
            for f in range(k + 1, len(vector_x)):
                sum_2 += matrix_w[f][k] * vector_prev[f]
            net = sum_1 + sum_2

            y = function(vector_prev[k], net)
            y_current.append(y)

        vector_prev = vector_y
        vector_y = y_current

    return vector_y


if __name__ == "__main__":
    patterns = [vectorized_pattern_1, vectorized_pattern_2, vectorized_pattern_3]
    matrix = get_matrix(patterns)
    test = [1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1]
    print_pattern(test)
    Y = recover_RNN(test, vectorized_pattern_3, matrix)
    print(Y)