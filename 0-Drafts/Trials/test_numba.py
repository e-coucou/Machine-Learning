import numpy as np
import numba as nb


import timeit
import sys
import random

np_v1 = np.array([2.35,5.56])
np_v2 = np.array([-3.35,1.56])
v1 = [2.35,5.56]
v2 = [-3.35,1.56]
arr = np.arange(10000)


def pure_add():
    return v1+v2

def numpy_add():
    return np.add(np_v1,np_v2)

def fonction(t=1000):
    acc = 0
    for i in range(t):
        x = random.random()
        y=random.random()
        if (x**2 + y**2) < 1.0 :
            acc += 1
    return acc *4 / t

def test_numpy():
    ret_arr = []
    for e in arr:
        if e%2 == 0:
            ret_arr.append(2)
        else :
            ret_arr.append(1)
    return ret_arr

@nb.vectorize
def scalar(num):
    if num%2==1:
        return 2
    else:
        return 1

def scalar_arr():
    scalar(arr)

@nb.vectorize("float64()",forceobj=True)
def numba_add():
    a = (np_v1,np_v2)


def main():
    # Essai
    n = 10000
    
    t1 = timeit.timeit(pure_add, number = n)
    print('Pure Python:', t1)
    t2 = timeit.timeit(numpy_add, number = n)
    print('Numpy:', t2)

    print('test',numpy_add())
    # t0 = timeit.timeit(numba_add,number = n)
    # print('Numba = ',t0)


    # f2 = nb.jit()(fonction)
    # t1 = timeit.timeit(f2,number=n)

    # print('temps = ',t1)

    t = timeit.timeit(test_numpy,number=1000)
    print('time en numpy', t)
    f = nb.njit()(test_numpy)
    t0 = timeit.timeit(f,number=1000)
    t = timeit.timeit(f,number=1000)
    print('time njit', t0,t)

    t0 = timeit.timeit(scalar_arr,number=1000)
    t = timeit.timeit(scalar_arr,number=1000)
    print('time vectorized', t0,t)


if __name__ == '__main__':
    sys.exit(main())