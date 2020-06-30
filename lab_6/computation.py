import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice

'''
Функция получает и записывает json-файл
:param url: ссылка
'''
def get_json(url):
    items = requests.get(url)
    data = items.json()
    with open('parking.json', 'w', encoding='utf-8') as write_file:
        json.dump(data, write_file, ensure_ascii=False, indent=4)


'''
Функция загружает сохранённый json-файл
:param filename: имя файла
:param return: данные
'''
def load_json(filename):
    with open(filename, 'r') as read_file:
        data = json.load(read_file)
    return data


'''
Функция выдаёт выборку вместительностей
:param n: число объектов
:param json_data: данные
:param return: массив вместительностей
'''
def get_capacities(n, json_data):
    prk = json_data['features']
    n_elements = list(islice(prk, n))

    capacity_list = list()
    for parking in n_elements:
        capacity_list.append(parking['properties']['Attributes']['CarCapacity'])
    return capacity_list


'''
Функция вычисляет расстояние между объектом и центром
:param weight: вес 
:param value: значение вместительности 
:param return: расстояние (разница)
'''
def get_diff(weight, value):
    return value - weight


'''
Функция обрабатки матрицы (победитель получает всё)
:param array: строка матрицы 
:param index: минимальный индекс 
'''
def render(array, index):
    for i in range(0, index):
        array[i] = 0
    array[index] = 1
    for i in range(index+1, len(array)):
        array[i] = 0


'''
Функция построения диаграммы по матрице распределения
:param matrix: матрица распределения
'''
def get_diagram(matrix):
    sizes = list()
    indexes = list()
    for index, line in enumerate(matrix):
        indexes.append(index)
        sizes.append(len(line))
    plt.grid()
    plt.bar(indexes, sizes)
    plt.xlabel('Кластеры')
    plt.ylabel('Количество парковок в кластере')
    plt.show()



