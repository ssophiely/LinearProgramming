import sys

import numpy as np
import constants as c
from exceptions import CountException


def resource_analysis(sol):
    n = c.a.shape[0] - 1
    status = [
        "Недефицитный" if float(
            sol[i]) > 0 else "Дефицитный" for i in range(
            n, 2 * n)]

    print("\nАнализ статуса ресурсов:")
    for i in range(n):
        print("Р1 - x{} - {} - {}".format(i + n + 1, sol[i + n], status[i]))

    print("\nАнализ ценности ресурсов:")
    for i in range(n):
        print("x{} - {}".format(i + n + 1, c.a[n][i + n]))


def print_solution():
    # s = [str(c.bi[c.basis_vector.index(i + 1)]) if (i + 1)
    #      in c.basis_vector else '0' for i in range(c.a.shape[1])]
    # print("\nОптимальное решение X=({})".format(", ".join(s)))
    # print("Значение целевой функции ", c.bi[c.a.shape[0] - 1])
    # return s
    print(c.a[-1,:], c.bi[-1])


def find_simplex():
    rows, columns = c.a.shape
    fake_count = rows - 2
    free_count = columns - fake_count
    var_l = ["x{}".format(i + 1) for i in range(free_count)] + \
        ["y{}".format(i + 1) for i in range(fake_count)]

    count = 1
    print("\nИсходная таблица ", count)
    print_m()

    # Шаг 1 - Проверка текущего допустимого базисного решения на оптимальность
    while True:
        rows, columns = c.a.shape
        # Шаг 2 - Нахождение разрешающего столбца
        p = list(c.a[rows - 1, :]).index(min(c.a[rows - 1, :]))

        # Шаг 3 - Проверка признака неограниченности целевой функции
        if np.all(c.a[:, p] <= 0):
            return print_solution()

        # Шаг 4 - Вычисление симплексных отношений
        si = [c.bi[i] / c.a[:, p][i]
              if c.a[:, p][i] > 0 else sys.maxsize for i in range(rows)]

        # Шаг 5 - Определение разрешающей строки
        q = si.index(min(si))

        # Шаг 6 - Переход к новому базису
        if c.basis_vector[q].startswith("y"):
            ind = var_l.index(c.basis_vector[q])
            var_l.pop(ind)
            c.a = np.delete(c.a, ind, 1)
        c.basis_vector[q] = var_l[p]

        rows, columns = c.a.shape
        # Шаг 7 - Нахождение нового допустимого базисного решения
        x = c.a[q][p]
        c.a[q] = np.array([c.a[q][i] / x for i in range(columns)])
        c.bi[q] = c.bi[q] / x

        for i in range(rows):
            if i != q:
                x = -c.a[i][p]
                c.bi[i] = c.bi[i] + c.bi[q] * x
                c.a[i] = np.array([c.a[i][j] + c.a[q][j] *
                                   x for j in range(columns)])

        print("\nИтерация ", count)
        print_m()
        count += 1

        if np.all(c.a[rows - 1, :] >= 0):
            if rows == fake_count + 2:
                c.a = np.delete(c.a, rows - 1, 0)
                c.bi.pop()
            else:
                return print_solution()


def print_m():
    ma, mb = c.a.shape  # размер исходной матрицы
    fake_count = ma - 2
    free_count = mb - fake_count

    print("Базис".ljust(9), end='')
    for i in range(free_count):
        print("x{}".format(i + 1).ljust(9), end='')
    for i in range(fake_count):
        print("y{}".format(i + 1).ljust(9), end='')
    print("\t\t" + "bi")

    for i in range(ma):
        if i < fake_count:
            print(c.basis_vector[i].ljust(9), end='')
        else:
            print("Z".ljust(9), end='')
        for j in range(mb):
            print(str('%.3f' % c.a[i, j]).ljust(9), end='')
        print("|\t" + str('%.3f' % c.bi[i]))


def matrix_init():
    a, b = input("Введите размер матрицы [i,j] через пробел   ").split()
    a, b = int(a), int(b)

    # Инициализация матрицы
    print("Введите коэффициенты матрицы ограничений через пробел")
    l = []
    for i in range(a):
        list_k = list(map(lambda x: float(x), input().split()))
        if len(list_k) != b:
            raise CountException
        l.append(list_k)
    c.a = np.array(l)

    # Добавление искусственных переменных
    for i in range(a):
        c.basis_vector.append("y{}".format(i + 1))
        basis = [[0] for _ in range(a)]
        basis[i][0] = 1
        c.a = np.append(c.a, basis, axis=1)

    # Матрица свободных членов
    print("Введите свободные члены через пробел")
    c.bi = list(map(lambda x: float(x), input().split()))
    if len(c.bi) != a:
        raise CountException

    # Коэффициенты целевой функции
    print("Введите коэффициенты целевой функции через пробел")
    Z = list(map(lambda x: float(x), input().split()))
    if len(Z) != b:
        raise CountException

    # Добавление двух оценочных строк
    c.a = np.append(c.a, [[-i for i in Z] + [0] * a], axis=0)
    c.a = np.append(c.a, [[-sum(c.a[:2, i])
                    for i in range(b)] + [0] * a], axis=0)
    c.bi.extend([0, -sum(c.bi)])
