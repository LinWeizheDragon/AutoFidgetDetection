import numpy as np
import math


def get_dynamics(X_0th, time=0.1):
    '''
    Compute Dynamics of X_0th
    :param X_0th:
    :param time: time delta
    :return:
    '''
    X_1st = np.zeros((X_0th.shape[0]-1, X_0th.shape[1]))
    X_2nd = np.zeros((X_0th.shape[0]-2, X_0th.shape[1]))
    for i in range(X_0th.shape[0]-1):
        X_1st[i] = (X_0th[i+1] - X_0th[i]) / time
    for j in range(X_0th.shape[0]-2):
        X_2nd[j] = (X_1st[j+1] - X_1st[j]) / time

    return np.hstack((X_0th[2:], X_1st[1:], X_2nd))


def get_CCC(X,Y):
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    assert X.shape[1] == 1
    assert Y.shape[1] == 1
    # print(np.hstack((X, Y)).T.shape)
    cov_mat = np.cov(np.hstack((X, Y)).T.astype(float))

    var_1 = math.sqrt(cov_mat[0, 0])
    var_2 = math.sqrt(cov_mat[1, 1])
    var_12 = cov_mat[1, 0]
    mean_1 = np.mean(X)
    mean_2 = np.mean(Y)
    #print(mean_1, mean_2)
    #print(var_1, var_2, var_12)

    return 2* var_12/(math.pow(var_1, 2) + math.pow(var_2, 2) + math.pow(mean_1 - mean_2, 2))
