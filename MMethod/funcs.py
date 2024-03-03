import sys

import numpy as np
import constants as c
from exceptions import CountException
from decimal import *

getcontext().prec = 3


def print_solution():
    var_l = ["x{}".format(i + 1) for i in range(c.free_count)] + \
            ["y{}".format(i + 1) for i in range(c.fake_count)]
    m_sol = [c.bi[c.basis_vector.index(
        i)] if i in c.basis_vector else 0 for i in var_l]
    d = dict(zip(var_l, m_sol))
    print(
        "\nОптимальное решение М-задачи: X=({})".format(
            ", ".join(
                "{} = {}".format(
                    k,
                    v) for k,
                v in d.items())))

    fake_vals = [d[k] for k in d.keys() if k.startswith("y")]
    if all([x == 0 for x in fake_vals]):
        print(
            "Оптимальное решение исходной задачи: X=({})".format(
                ", ".join(
                    "{} = {}".format(
                        k,
                        v) for k,
                    v in d.items() if k.startswith("x"))))
    else:
        print(
            "Система ограничений исходной задачи несовместна в области допустимых решений")


def find_simplex():
    var_l = ["x{}".format(i + 1) for i in range(c.free_count)] + \
        ["y{}".format(i + 1) for i in range(c.fake_count)]

    count = 1
    print("\nИсходная таблица ")
    print_m(var_l)

    # Шаг 1 - Проверка текущего допустимого базисного решения на оптимальность
    while True:
        rows, columns = c.a.shape
        # Шаг 2 - Нахождение разрешающего столбца
        p = list(c.a[rows - 1, :]).index(min(c.a[rows - 1, :]))

        # Шаг 3 - Проверка признака неограниченности целевой функции
        if np.all(c.a[:, p] <= 0):
            return print("М-задача неразрешима")

        # Шаг 4 - Вычисление симплексных отношений
        si = [c.bi[i] / c.a[:, p][i]
              if c.a[:, p][i] > 0 else sys.maxsize for i in range(rows)]

        # Шаг 5 - Определение разрешающей строки
        q = si.index(min(si))

        # Шаг 6 - Переход к новому базису
        v = c.basis_vector[q]
        c.basis_vector[q] = var_l[p]
        if v.startswith("y"):
            ind = var_l.index(v)
            var_l.pop(ind)
            c.a = np.delete(c.a, ind, 1)

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
        print_m(var_l)
        count += 1

        if np.all(c.a[rows - 1, :] >= 0):
            if rows == c.fake_count + 2:
                c.a = np.delete(c.a, rows - 1, 0)
                c.bi.pop()
            else:
                return print_solution()


def print_m(l):
    ma, mb = c.a.shape  # размер исходной матрицы

    print("Базис".ljust(9), end='')
    for i in range(mb):
        print(l[i].ljust(9), end='')
    print("\t\t" + "bi")

    for i in range(ma):
        if i < c.fake_count:
            print(c.basis_vector[i].ljust(9), end='')
        else:
            print("Z".ljust(9), end='')
        for j in range(mb):
            print(str(c.a[i, j]).ljust(9), end='')
        print("|\t" + str(c.bi[i]))


def matrix_init():
    a, b = input("Введите размер матрицы [i,j] через пробел   ").split()
    a, b = int(a), int(b)
    c.fake_count = a
    c.free_count = b

    # Инициализация матрицы
    print("Введите коэффициенты матрицы ограничений через пробел")
    l = []
    for i in range(a):
        list_k = list(map(lambda x: Decimal(x), input().split()))
        if len(list_k) != b:
            raise CountException
        l.append(list_k)
    c.a = np.array(l, dtype=Decimal)

    # Добавление искусственных переменных
    for i in range(a):
        c.basis_vector.append("y{}".format(i + 1))
        basis = [[Decimal(0)] for _ in range(a)]
        basis[i][0] = Decimal(1)
        c.a = np.append(c.a, basis, axis=1)

    # Матрица свободных членов
    print("Введите свободные члены через пробел")
    c.bi = list(map(lambda x: Decimal(x), input().split()))
    if len(c.bi) != a:
        raise CountException

    # Коэффициенты целевой функции
    print("Введите коэффициенты целевой функции через пробел")
    Z = list(map(lambda x: Decimal(x), input().split()))
    if len(Z) != b:
        raise CountException

    # Добавление двух оценочных строк
    c.a = np.append(c.a, [[-i for i in Z] + [Decimal(0)] * a], axis=0)
    c.a = np.append(c.a, [[Decimal(-sum(c.a[:2, i])) for i in range(b)] + [Decimal(0)] * a], axis=0)
    c.bi.extend([Decimal(0), -sum(c.bi)])
