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

'''
-3 0 2 3 3 4
3 -2 1 4 3 -3
4 -2 1 -4 -1 -1
2 3 3 1 2 -3

1 2 2 3

4 4 2 4 3 1
'''