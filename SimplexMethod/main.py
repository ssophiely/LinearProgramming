from funcs import *
from exceptions import *

try:
    matrix_init()
    solution = simplex_method()
    resource_analysis(solution)
except CoefException:
    print("Введено неверное количество значений")


'''
2 -1 1
-4 2 -1
3 0 1
'''