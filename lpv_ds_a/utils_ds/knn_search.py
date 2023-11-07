import numpy as np


def knn_search(Mu, att, size):
    distances = np.zeros(len(Mu))
    index = 0
    for mu in Mu:
        distances[index] = np.linalg.norm(mu - att)
        index = index + 1

    order = []
    for i in np.arange(0, len(Mu)):
        cur_value = distances[i]
        cur_index = 0
        for j in np.arange(0, len(Mu)):
            if distances[j] < cur_value:
                cur_index += 1
        order.append(cur_index)

    return order

