from funcs import *
from exceptions import *

try:
    matrix_init()
    find_simplex()
except CountException:
    print("Введено неверное количество значений")


'''
1 3 2 2
2 2 1 1

5 3 4 -1
'''