from computation import *
from prettytable import PrettyTable

'''
Основная функция кластеризации
:param capacities: вместительности парковок
:param centers: центры кластеров
:param return: матрица весов, матрица распределения на кластеры
'''
def clusterize(capacities, centers):
    matrix = np.zeros((len(capacities), len(centers)))
    for i, value in enumerate(capacities):
        for j, center in enumerate(centers):
            matrix[i][j] = np.abs(value - center)
    clusterized = [[] for x in range(len(centers))]
    for i in range(len(matrix)):
        index = np.argmin(matrix[i])
        clusterized[index].append(capacities[i])
        render(matrix[i], index)
    return matrix, clusterized



if __name__ == "__main__":
    # get_json("https://apidata.mos.ru/v1/datasets/623/features?api_key=e7001de51ff1e03bfe9a57548d25075b")
    data = load_json('parking.json')
    capacities = get_capacities(200, data)
    centers = [1, 5, 10, 15, 20, 25, 30, 35]
    matrix, result = clusterize(capacities, centers)

    table = PrettyTable(['№ кластера', 'Центр', 'Вместимости парковок'])
    for index, (line, weight) in enumerate(zip(result, centers)):
        table.add_row([index, np.round(weight, 4), line])
    print(table)
    get_diagram(result)
    print(capacities)
