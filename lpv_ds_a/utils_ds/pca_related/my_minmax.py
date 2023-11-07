import numpy as np


def my_minmax(x):
    x_sorted = np.sort(x)
    mm = np.array([x_sorted[0], x_sorted[-1]])
    return mm


if __name__ == '__main__':
    x = np.array([1,2,3,4,4,2,1])
    print(x[x>2])
    a = 5
    print(np.in1d(a, x))
    print(my_minmax(x))
    # test code
