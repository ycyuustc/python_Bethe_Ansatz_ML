import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import multiprocessing as mp
import time
import pickle
import os

# global variable path
MY_PATH = 'E:/pythondata/'
FILE_HEADER = 'Num_10'

Num_iteration = 200
Num0 = 10
Num_thread = 4
Num = Num0 + 1

list_X = list()
list_y = list()
for index in range(Num_thread):
    for iIteration in range(Num_iteration):
        file_name = (MY_PATH + FILE_HEADER + '_FULL_' + 'th_' + str(index)
                     + '_it_' + str(iIteration))
        with open(file_name, 'rb') as File:
            mylist = pickle.load(File)
        mylist_1 = mylist[0]
        mylist_2 = mylist[1]
        for iInd in range(Num):
            list_X.append(mylist_1[iInd])
            v_root = mylist_2[iInd]
            myenergy = np.sum(v_root ** 2)
            list_y.append(myenergy)

ba_full_X = np.array(list_X)
ba_full_y = np.array(list_y)

list_X = list()
list_y = list()
for index in range(Num_thread):

    file_name = MY_PATH + FILE_HEADER + '_th_' + str(index)
    with open(file_name, 'rb') as File:
        mylist = pickle.load(File)
    mylist_1 = mylist[0]
    mylist_2 = mylist[1]
    for iIteration in range(Num_iteration):
        list_X.append(mylist_1[iIteration])
        v_root = mylist_2[iIteration]
        myenergy = np.sum(v_root ** 2)
        list_y.append(myenergy)

ba_X = np.array(list_X)
ba_y = np.array(list_y)

file_name = (MY_PATH + FILE_HEADER + '_ba_Xy')
with open(file_name, 'wb') as myfile:
    mylist = [ba_X, ba_y]
    pickle.dump(mylist, myfile)

file_name = (MY_PATH + FILE_HEADER + '_ba_FULL_Xy')
with open(file_name, 'wb') as myfile:
    mylist = [ba_full_X, ba_full_y]
    pickle.dump(mylist, myfile)