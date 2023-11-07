import numpy as np
from scipy.stats import multivariate_normal


def regress(data, assignmentArray):
    K = np.max(assignmentArray)+1
    # print(K)
    clusterList = []
    for k in range(K):
        data_k = data[assignmentArray==k]
        mean_k = np.mean(data_k, axis=0)
        cov_k = np.cov(data_k.T)
        clusterList.append(multivariate_normal(mean=mean_k, cov=cov_k, allow_singular=True))
        # clusterList.append((mean_k, cov_k))

    regressedAssignmentArray = np.zeros((data.shape[0]), dtype=int)
    for i in range(data.shape[0]):
        probArray = np.zeros((K, ))
        for k in range(K):
            probArray[k] = clusterList[k].pdf(data[i, :])
        regressedAssignmentArray[i] = np.argmax(probArray)


    # print(regressedAssignmentArray)
    return regressedAssignmentArray