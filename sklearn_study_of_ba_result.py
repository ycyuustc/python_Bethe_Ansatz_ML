import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
from sklearn.svm import SVR

import random as rnd
import multiprocessing as mp
import time
import pickle
import os

# global variable path
MY_PATH = 'E:/pythondata/'
FILE_HEADER = 'Num_1000_n0_ba_Xy'

file_name = MY_PATH + FILE_HEADER

with open(file_name, 'rb') as myfile:
    mylist = pickle.load(myfile)

m_ind = mylist[0]
v_energy = mylist[1]

ba_X = m_ind
ba_y = v_energy

# 因为是蒙卡的原因，需要扔掉一些点。这里扔掉每个seed产生的
# 前 Num_drop = 10 个点
# 生成的 tba_X 和 tba_y 是去掉这些点后重新排布得到的训练集

Num0 = 1000
Num = Num0 + 1
Num_thread = 20
Num_iteration = 200

Num_drop = 10

tba_X = np.zeros([(Num_iteration - Num_drop)*Num_thread, Num])
tba_y = np.zeros([(Num_iteration - Num_drop)*Num_thread])

for i1 in range(0, Num_iteration - Num_drop):
    for i2 in range(0, Num_thread):
        X = ba_X[i2*Num_iteration + i1 + Num_drop, :]
        y = ba_y[i2*Num_iteration + i1 + Num_drop]
        tba_X[i1*Num_thread + i2, :] = X
        tba_y[i1*Num_thread + i2] = y

# 主要用于求解BA方程的代码

def fn_fj(vx, v_ind, c):
    num = vx.size
    mj = np.matmul(np.ones([num, 1]), vx.reshape(1, num))
    ml = np.matmul(vx.reshape(num, 1), np.ones([1, num]))
    v_result = -2 * np.pi * v_ind + vx + 2 * np.sum(np.arctan(1 / c * (mj - ml)), axis=0)
    return v_result


def fn_m(vx, c):
    num = vx.size
    mj = np.matmul(np.ones([num, 1]), vx.reshape(1, num))
    ml = np.matmul(vx.reshape(num, 1), np.ones([1, num]))
    m_r = 2 * c / (c ** 2 + (mj - ml) ** 2)
    v_r = np.sum(m_r, axis=0) + np.ones(num)
    tm1 = np.diag(v_r)
    tm2 = -m_r

    m_result = tm1 + tm2
    return m_result


def fn_steepest(vx, v_ind, lamb):
    c = 1.0 / lamb

    while 1:

        dfj = fn_fj(vx, v_ind, c)
        matrix = fn_m(vx, c)
        dvx = -np.linalg.solve(matrix, dfj)
        vx = vx + dvx

        if np.linalg.norm(fn_fj(vx, v_ind, c)) < 1e-8:
            break

    return vx


def ba_solve_liebliniger_large_c(v_ind, coup):
    v0 = 2 * np.pi * v_ind
    vx = fn_steepest(v0, v_ind, coup)

    return vx


test_ind = 3422
coupling = 0.0002
num0 = 1000
v_base = 1.0 / 2 * np.arange(-num0, num0 + 1.0, step=2)
v_ind_onehot = tba_X[test_ind, :]
v_ind = v_base[v_ind_onehot > 0.5].copy()

v_root = ba_solve_liebliniger_large_c(v_ind, coupling)

print(tba_y[test_ind])
print(np.sum(v_root**2))

from sklearn.cross_validation import cross_val_score
gamma_range = np.linspace(1e-4, 2e-4, 20)
g_score = []
for gamma in gamma_range:
    svr_rbf = SVR(kernel='rbf', C=1e10, gamma=gamma)
    scores = cross_val_score(svr_rbf, tba_X[0:400], tba_y[0:400], cv=5) # , scoring='mean_squared_error')
    print('gamma=' + str(gamma))
    print('score=' + str(scores.mean()))
    print('====================================================')
    g_score.append(scores.mean())

print(gamma_range)
print(g_score)


