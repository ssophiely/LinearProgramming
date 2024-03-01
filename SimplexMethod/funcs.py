import sys

import numpy as np
import constants as c


def resource_analysis(sol):
    n = c.a.shape[0] -1
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
    s = [str(c.bi[c.basis_vector.index(i + 1)]) if (i + 1)
         in c.basis_vector else '0' for i in range(c.a.shape[1])]
    print("\nОптимальное решение X=({})".format(", ".join(s)))
    print("Значение целевой функции ", c.bi[c.a.shape[0] - 1])
    return s


def simplex_method():
    rows, columns = c.a.shape
    count = 1
    print("\nИсходная таблица ", count)
    print_m()

    # Шаг 1 - Проверка текущего допустимого базисного решения на оптимальность
    while np.any(c.a[rows - 1, :] < 0):
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
        c.basis_vector[q] = p + 1

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
    else:
        return print_solution()


def print_m():
    rows, columns = c.a.shape

    print("Xki".ljust(9), end='')
    for i in range(columns):
        print("x{}".format(i + 1).ljust(9), end='')
    print("\t\t" + "bi")

    for i in range(rows):
        if i == rows - 1:
            print("Z".ljust(9), end='')
        else:
            print("x{}".format(c.basis_vector[i]).ljust(9), end='')
        for j in range(columns):
            print(str('%.3f' % c.a[i, j]).ljust(9), end='')
        print("|\t" + str('%.3f' % c.bi[i]))


def matrix_init():
    size = input("Введите количество неизвестных x   ")
    n = int(size)

    # Инициализация матрицы
    print("Введите коэффициенты матрицы ограничений через пробел")
    a = []
    for i in range(n):
        list_k = list(map(lambda x: float(x), input().split()))
        if len(list_k) != n:
            raise CoefException
        a.append(list_k)
    c.a = np.array(a)

    # Добавление базисных переменных
    for i in range(n):
        c.basis_vector.append(i + n + 1)
        basis = [[0] for _ in range(n)]
        basis[i][0] = 1
        c.a = np.append(c.a, basis, axis=1)

    # Матрица свободных членов
    print("Введите свободные члены через пробел")
    c.bi = list(map(lambda x: float(x), input().split()))
    if len(c.bi) != n:
        raise CoefException

    # Коэффициенты целевой функции
    print("Введите коэффициенты целевой функции через пробел")
    Z = list(map(lambda x: float(x), input().split()))
    if len(Z) != n:
        raise CoefException
    c.a = np.append(c.a, [Z + [0] * n], axis=0)
    c.bi.append(0)
