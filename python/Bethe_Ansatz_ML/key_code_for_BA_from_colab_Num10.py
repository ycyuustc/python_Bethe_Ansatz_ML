import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import multiprocessing as mp
import time
import pickle
import os
# from google.colab import files

# ###### import scipy.io as sio


# global variable path
MY_PATH = 'E:/pythondata/'
FILE_HEADER = 'Num_10'


# =============================  functions  =====================================
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


# save the result into the /data

def my_ba_thread(inpdata):
    global v_root
    #  The timer for each thread
    time_begin = time.time()

    num0 = inpdata[0]
    coupling = inpdata[1]
    beta = inpdata[2]
    mu = inpdata[3]
    num_iteration = inpdata[4]
    index = inpdata[5]

    num = num0 + 1

    # alpha in the grand canonical ensemble
    alpha = -beta * mu

    v_base0 = 1.0 / 2 * np.arange(-num0, num0 + 1.0, step=2)
    # The spectrum at zero temperature
    vx = 2.0 * np.pi * v_base0
    v_energy = vx ** 2
    v_prob = 1 / (1 + np.exp(beta * (v_energy - mu)))

    v_base = v_base0.copy()

    v_onehot = np.float32((v_prob
                           - np.array([rnd.uniform(0, 1) for _ in range(num)])) > 0)

    m_ind_par = list()
    c_root_par = list()

    # ==============    Main Loop    =======================================
    for iIteration in range(num_iteration):
        time_1 = time.time()

        m_ind_par_full = list()
        c_root_par_full = list()
        for iInd in range(num):

            v_onehot1 = v_onehot.copy()
            v_onehot1[iInd] = 0.0
            v_onehot2 = v_onehot.copy()
            v_onehot2[iInd] = 1.0

            v_ind1 = v_base[v_onehot1 > 0.5].copy()
            v_ind2 = v_base[v_onehot2 > 0.5].copy()

            v_root1 = ba_solve_liebliniger_large_c(v_ind1, coupling)
            v_root2 = ba_solve_liebliniger_large_c(v_ind2, coupling)

            shoulder1 = beta * np.sum(v_root1 ** 2) + alpha * np.sum(v_onehot1)
            shoulder2 = beta * np.sum(v_root2 ** 2) + alpha * np.sum(v_onehot2)

            p1 = 1.0 / (1.0 + np.exp(shoulder1 - shoulder2))

            if rnd.uniform(0, 1) < p1:
                v_onehot = v_onehot1
                v_root = v_root1
            else:
                v_onehot = v_onehot2
                v_root = v_root2

            m_ind_par_full.append(v_onehot)
            c_root_par_full.append(v_root)
        instore_full = [m_ind_par_full, c_root_par_full, inpdata]
        thread_data_name_full = (MY_PATH + FILE_HEADER + '_FULL_' + 'th_' + str(index)
                                 + '_it_' + str(iIteration))
        with open(thread_data_name_full, 'wb') as file:
            pickle.dump(instore_full, file)

        m_ind_par.append(v_onehot)
        c_root_par.append(v_root)

        time_2 = time.time()
        print('No.' + str(index) + ' thread' + ' iteration ' + str(iIteration)
              + ' finished' + ' cost time: ' + str(time_2 - time_1))

    instore = [m_ind_par, c_root_par, inpdata]

    #   ==========   restore the result  =====================
    # thread_data_name_full = MY_PATH + FILE_HEADER + '_FULL_' + str(index)  # pickle it into the root
    thread_data_name = MY_PATH + FILE_HEADER + '_th_' + str(index)
    # with open(thread_data_name_full, 'wb') as File:
    #     pickle.dump(instore_full, File)
    with open(thread_data_name, 'wb') as File:
        pickle.dump(instore, File)

    time_end = time.time()
    print('The thread ' + str(index) + ' takes '
          + str(time_end - time_begin) + ' seconds')

    # return [m_ind_par, c_root_par, v_base0]  need return value ??


# ===================== The thread =========================
def multicore(num0, num_thread, num_iteration, coupling, beta, mu):
    # ==========  SET UP the thread  ==============================
    # num_thread = 40                      # The number of thread
    # num0 = 10                            # The total number -1
    # num_iteration = 200                  # The total number for iteration in each thread
    # coupling :               The inverse of c
    # beta = 0.000005 :        The inverse temperature
    # mu = ((num0 * 0.5) * np.pi) ** 2 :        Thechemical potential

    listparameter = (num0, coupling, beta, mu, num_iteration)

    inpdata = [listparameter + (i2,) for i2 in range(num_thread)]

    pool = mp.Pool()
    pool.map(my_ba_thread, inpdata)

    # return result ?  need return value


# =============== MAIN PROGRAM =============================================
if __name__ == '__main__':

    #  SETTINGS
    # Num_thread = 4                                    # The number of thread
    # Num0 = 1000                                       # The total number -1
    # Num_iteration = 5                                 # The total number for iteration
    #                                                        in each thread
    # file_Header = 'gdndsb'                            # header for the file name
    # coupling = 0.0002                                 # The inverse of c
    # beta = 0.000005                                   # The inverse temperature
    # mu = ((num0 * 0.5) * np.pi) ** 2                  # The chemical potential

    # =====  create the directory and file in cloud to store datas   ==========
    to_make_dir = MY_PATH
    if os.path.exists(to_make_dir):
        print('The directory %s has already existed' % to_make_dir)
    else:
        print('Try to create the %s' % to_make_dir)
        os.mkdir(to_make_dir)

    Num0 = 10
    Num_thread = 4
    Num_iteration = 200
    st0 = time.time()
    multicore(num0=Num0,
              num_thread=Num_thread,
              num_iteration=Num_iteration,
              coupling=0.05,
              beta=0.05,
              mu=((Num0 * 0.30) * np.pi) ** 2
              )
    st1 = time.time()
    print('Lapse time = ', st1 - st0)
    # =============   Main calculation finished ===============================

    # =============  Deal with the data  in /data/=============================
    # Num0 =
    # Num = Num0 + 1
    # Num_thread =

    Num = Num0 + 1
    v_Base0 = 1.0 / 2 * np.arange(-Num0, Num0 + 1.0, step=2)

    m_distribution = np.zeros([Num_thread, Num])

    for index in range(Num_thread):
        myFileName = MY_PATH + FILE_HEADER + '_th_' + str(index)
        with open(myFileName, 'rb') as file_read:
            mylist = pickle.load(file_read)
            myarray = np.array(mylist)
        result_index = myarray[0]
        result_index = np.mean(result_index, axis=0)
        v_distribution = np.array(result_index)
        m_distribution[index, :] = v_distribution
        # print(v_distribution)
        # fig = plt.figure()
        # plt.plot(v_Base0, v_distribution, c='r', marker='.')

    v_dist = np.mean(m_distribution, axis=0)
    # fig = plt.figure()
    plt.plot(v_Base0, v_dist, c='b', marker='*')

    plt.show()
    plt.close()

    # save the result as soon as possible.

    #  for i in range(Num_thread):
    # print('begin thread' + str(i))
    # t1 = time.time()
    # files.download(MY_PATH + FILE_HEADER + str(i))
    # t2 = time.time()
    # print('finished the No.' + str(i) + ' thread, Lapse time = ', str(t2-t2))

# make up the training set.
# ba_X presents the input
# ba_y presents the output

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
    with open(file_name, 'wb') as file:
        mylist = [ba_X, ba_y]
        pickle.dump(mylist, file)

    file_name = (MY_PATH + FILE_HEADER + '_ba_FULL_Xy')
    with open(file_name, 'wb') as file:
        mylist = [ba_full_X, ba_full_y]
        pickle.dump(mylist, file)



