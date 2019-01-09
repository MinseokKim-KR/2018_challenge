import gzip
import os
import sys
import urllib

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

# BASE = "../autron/driver_data_example"
BASE = "../autron/driver_data"

i=0
c=0

Dates = ['170217', '170220', '170518', '170530', '170601']
Name = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11']
Test = ['test1', 'test2', 'test3']
# col = [i+row for row in range(0,438)]
Driver = ['TW', 'JH', 'HH', 'CS']
# Dates = ['170217']
# Name = ['L2']
# Test = ['test1']
color = ['b','r','y','g']



# # tf.app.flags.DEFINE_string('resf', RESF, "Set the result file name")
#
# tf.app.flags.DEFINE_string('Driver', Driver, "Set")
# tf.app.flags.DEFINE_string('Driver', Dates, "Set")
# tf.app.flags.DEFINE_string('Driver', Name, "Set")
# tf.app.flags.DEFINE_string('Driver', Test, "Set")
#
# # tf.app.flags.DEFINE_integer('train_size',   TRAIN_SIZE, "Set")
# FLAGS = tf.app.flags.FLAGS
#
#
#
# Driver = FLAGS.Driver
# # Dates = ['170217', '170220', '170518', '170530', '170601']
# # Name = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11']
# Dates = FLAGS.Dates
# Name = FLAGS.Name
# Test = FLAGS.Test
#
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

# table = [TW,JH,HH,CS]
table = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
# choose = [cTW,cJH,cHH,cCS]
choose = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
temp = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
m=[0.0,0.0,0.0,0.0]
v=[0.0,0.0,0.0,0.0]
gauss=[0.0,0.0,0.0,0.0]
#read all data and append table

def make_gauss(N, sig, mu):
    # if sig ==0:
    #     return
    # else:
    #     return lambda x: N/(sig*(2*np.pi)**.5) * np.e **(-(x-mu)**2/(2*sig**2))
    return lambda x: N / (sig * (2 * np.pi) ** .5) * np.e ** (-(x - mu) ** 2 / (2 * sig ** 2))



# for i in range(len(Dates)):
#     for j in range(len(Driver)):
#         c=0

def main_index(Driver= None, Dates=None, Test=None, color=None, table=None, choose=None, m=None, v=None, gauss=None, temp=None , Name = None, table2=None):
    Datess = [Dates]

    for i in range(len(Driver)):
        c=0
        # print len(Driver)
        for j in range(len(Datess)):
            # print len(Datess)
            # for k in range(len(Test)):
            #     # print len(Test)
            #     # dirName = BASE+'/'+Datess[i]+'/'+Driver[j]+'/'+ Datess[i]+'_'+Test[h]+'_'+Driver[j]+'/'
            #     for l in range(len(Name)):
            #         # print len(Name)
                    dirName = BASE + '/' + Datess[j] + '/' + Driver[i] + '/' + Datess[j] + '_' + Test + '_' + Driver[i] + '/'+Datess[j] + '_' + Test+ '_' + Driver[i] + '_' + Name+'.csv'
                    # print dirName
                    if os.path.exists(dirName):
                        print dirName
                        f = pd.read_table(dirName, sep=',')
                        if c != 0:
                            # table[i]=table[i].append(f,ignore_index=True)
                            table[i] = table[i].append(f)
                        else:
                            table[i]=f
                            c=1
    # dirName = BASE + '/' + Datess+ '/' + Driver + '/' + Datess + '_' + Test + '_' + Driver + '/' + \
    #           Datess + '_' + Test + '_' + Driver + '_' + Name + '.csv'
    # if os.path.exists(dirName):
    #     print dirName
    #     f = pd.read_table(dirName, sep=',')
    #     table[i] = f
    # np.savetxt(Driver + Dates + Name + '_0703', table, delimiter=',')
    print Name
    print Test
    index = table[0].axes

    for i in range(1, 438):
        for j in range(0, 4):
            if len(table[j]) != 0 :
                choose[j] = table[j][index[1][i]]
                choose[j] = choose[j].dropna()  # remove Nan
                table2[j] = pd.concat([table2[j],choose[j]], axis=1)

    print 'save start'
    for x in range(len(Driver)):
        if len(table[x]) != 0:
            for y in range(len(Datess)):
                    # for z in range(len(Name)):
                    #     for e in range(len(Test)):
                        np.savetxt(Driver[x]+'_'+Datess[y]+'_'+Name+'_'+Test + '_0703', table2[x], delimiter=',')
    table = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    table2 = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]



for i in range(len(Dates)):

    TW = pd.DataFrame()
    JH = pd.DataFrame()
    HH = pd.DataFrame()
    CS = pd.DataFrame()

    cTW = pd.DataFrame()
    cJH = pd.DataFrame()
    cHH = pd.DataFrame()
    cCS = pd.DataFrame() #clear
    table = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    table2 = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    # choose = [cTW,cJH,cHH,cCS]
    choose = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    temp = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    for a in range(len(Name)):
        print Name[a]
        for b in range(len(Test)):
            print Test[b]
            main_index(Driver=Driver, Dates=Dates[i], Test=Test[b], Name = Name[a],color=color, table=table, table2=table2, choose=choose, m=m, v=v, gauss=gauss, temp=temp)


