import csv
import os
import numpy
import numpy as np
import random
from joblib import Parallel, delayed
import multiprocessing

import pandas as pd
import os

# BASE = "../autron/driver_data_test"

BASE = "/home/mskim/autron/p_value/driver_data_1705182"

i=0
c=0
# col = [i+row for row in range(0,438)]
# Driver = ['TW', 'JH', 'HH', 'CS']
# Dates = ['170220']
# Test = ['test1', 'test2', 'test3']
# color = ['b','r','y','g']
#
# Driver = ['TW', 'JH', 'HH', 'CS']
# # Driver = ['TW', 'JH', 'KT', 'CS']
# Dates = ['170217', '170220', '170518', '170530', '170601']
# Name = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11']
# # Dates = ['170217']
# Test = ['test1', 'test2', 'test3']
# color = ['b','r','y','g']
#
#
# TW=pd.DataFrame()
# JH=pd.DataFrame()
# HH=pd.DataFrame()
# CS=pd.DataFrame()
#
# cTW=pd.DataFrame()
# cJH=pd.DataFrame()
# cHH=pd.DataFrame()
# cCS=pd.DataFrame()
# #
# # table = [TW,JH,HH,CS]
# # choose = [cTW,cJH,cHH,cCS]
# # table = [TW,JH,HH,CS]
# table = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
# choose = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
# temp = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
# m=[0,0,0,0]
# v=[0,0,0,0]
# gauss=[0,0,0,0]
# #read all data and append table
#
# def make_gauss(N, sig, mu):
#     return lambda x: N/(sig*(2*np.pi)**.5) * np.e **(-(x-mu)**2/(2*sig**2))
#
# # for i in range(len(Dates)):
# #     for j in range(len(Driver)):
# #         c=0
#
# for j in range(len(Driver)):
#     c=0
#     for i in range(len(Dates)):
#         for h in range(len(Test)):
#             dirName = BASE+'/'+Dates[i]+'/'+Driver[j]+'/'+ Dates[i]+'_'+Test[h]+'_'+Driver[j]+'/'
#             if os.path.exists(dirName):
#                 for fn in os.listdir(dirName):
#                     dirName1= os.path.join(dirName,fn)
#                     print dirName1
#                     for z in range(len(Driver)):
#                         print len(table[z])
#                     f = pd.read_table(dirName1,sep=',')
#                     if c!=0:
#                         table[j]=table[j].append(f,ignore_index=True)
#                     else:
#                         table[j]=f
#                         c=1
# index = table[0].axes
#
# user configuration
# BASE = './sensor10_driver'
# BASE = './extract_test_data'
MIN_ROWS = 32

Drivers = ['CS','HH','JH','TW']
X_width = 2
Labels = {}




def get_files(drivers=None, BASE = None):
    if drivers is None or len(drivers) == 0:
        drivers = Drivers

    filenames = []
    filelabels = []

    dirName = BASE
    for driver in drivers:
        dirName1 = os.path.join(dirName, driver)
        print("driver = %s" % (driver))

        if os.path.exists(dirName1):
            for fn in os.listdir(dirName1):
                filenames.append(dirName1 + '/' + fn)
                filelabels.append(Labels[driver])

    print "found %d files" % len(filenames)
    return filenames, filelabels


def fileReader(fn):
    reader = csv.reader(open(fn, "rb"), delimiter=',')
    x = list(reader)
    result = numpy.array(x).astype('float')
    print '%s has %d rows.' % (fn, result.shape[0])
    return result


# Batch Normalization.
def nomalization_matrix2(table,  WIDTH): #make input data to nomalization and reshape 300 * 2
    MAX = np.zeros(WIDTH)
    MIN = np.zeros(WIDTH) #set up the max and min of the two sensor
    x_data=[]
    # print "table"
    # print table.shape
    for i in range(WIDTH):
        for j in range(len(table)):
            MAX[i] = max(MAX[i], float(table[j][i]))
            MIN[i] = min(MIN[i], float(table[j][i])) #insert max and min
    temp = np.zeros(WIDTH)
    # temp = np.array(temp) #make temp and it will be train or test data
    for i in range(len(table)):
        for j in range(WIDTH):
            if MAX[j]-MIN[j] != 0 :
                temp[j] = float(float(table[i][j])- MIN[j])/float(MAX[j]-MIN[j])
            else :
                temp[j] = 0
        x_data = np.append(x_data, temp)
        temp = np.zeros(WIDTH)

    x_data = np.reshape(x_data, (-1, WIDTH))
    return x_data

# groups = None, personIDs = None,
def extract_data_oned(drivers=None, labels=None, numRows=None, data_types = None, numData=None, mode=None, NUM_CHANNELS=None,
                     ONED=True, DATA_SIZE=None, BASE=None):

    NUM_LABELS = len(set(labels))
    print NUM_LABELS
    for s, l in zip(drivers, labels):
        Labels[s] = l
    # Labels=['CS:0', 'HH:1','JH:2', 'TW:3']
    filenames, filelabels = get_files(drivers,BASE)

    results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(fileReader)(fn) for fn in filenames)

    x_data=numpy.ndarray(shape=(numData, numRows*NUM_CHANNELS), dtype=numpy.float32)

    if ONED:
        data = numpy.ndarray(shape=(numData, numRows, NUM_CHANNELS), dtype=numpy.float32)
    else:
        data = numpy.ndarray(shape=(numData, numRows, DATA_SIZE, 1), dtype=numpy.float32)
    dataLabel = numpy.zeros(shape=(numData), dtype=numpy.float64)

    numDataCount = 0
    while numDataCount < numData:
        res_idx = random.randint(0, len(results) - 1)
        max_row_idx = results[res_idx].shape[0] - numRows

        if max_row_idx < MIN_ROWS:  # this value should be at least 0
            continue

        if mode is 'train':
            row_idx = random.randint(0, max_row_idx * 4 / 5)  ## train : 4 / 5
        elif mode is 'validate':
            row_idx = random.randint(max_row_idx * 4 / 5, max_row_idx)  ## validate : 1 / 5
        if ONED:
            data[numDataCount, :, :] = results[res_idx][row_idx:row_idx + numRows, 0:NUM_CHANNELS]
            data[numDataCount, :, :] = nomalization_matrix2(data[numDataCount, :, :], DATA_SIZE)
            x_data[numDataCount,:] = data[numDataCount, :, :].reshape(-1)
        else:
            data[numDataCount, :, :, 0] = results[res_idx][row_idx:row_idx + numRows, 0:DATA_SIZE]
            data[numDataCount, :, :] = nomalization_matrix2(data[numDataCount, :, :], DATA_SIZE)
        filelabels[res_idx] = int(filelabels[res_idx])

        # dataLabel[numDataCount, filelabels[res_idx]] = 1.0
        dataLabel[numDataCount] = float(filelabels[res_idx])
        numDataCount = numDataCount + 1
        print numDataCount
    return x_data, dataLabel


    if __name__ == '__main__':
        extract_data_oned()



