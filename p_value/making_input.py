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
BASE = "../autron/driver_data_test"

i=0
c=0


# col = [i+row for row in range(0,438)]
Driver = ['TW', 'JH', 'HH', 'CS']
Dates = ['170217', '170220', '170518', '170530', '170601']
Name = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11']
# Dates = ['170217']
Test = ['test1', 'test2', 'test3']
color = ['b','r','y','g']


TW=pd.DataFrame()
JH=pd.DataFrame()
HH=pd.DataFrame()
CS=pd.DataFrame()

cTW=pd.DataFrame()
cJH=pd.DataFrame()
cHH=pd.DataFrame()
cCS=pd.DataFrame()

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

def main_index(Driver= None, Dates=None, Test=None, color=None, table=None, choose=None, m=None, v=None, gauss=None, temp=None ):
    Datess = [Dates]

    for i in range(len(Driver)):
        c=0
        # print len(Driver)
        for j in range(len(Datess)):
            # print len(Datess)
            for k in range(len(Test)):
                # print len(Test)
                # dirName = BASE+'/'+Datess[i]+'/'+Driver[j]+'/'+ Datess[i]+'_'+Test[h]+'_'+Driver[j]+'/'
                for l in range(len(Name)):
                    # print len(Name)
                    dirName = BASE + '/' + Datess[j] + '/' + Driver[i] + '/' + Datess[j] + '_' + Test[k] + '_' + Driver[i] + '/'+Datess[j] + '_' + Test[k] + '_' + Driver[i] + '_' + Name[l]+'.csv'
                    if os.path.exists(dirName):
                        print dirName
                        f = pd.read_table(dirName, sep=',')
                        if c != 0:
                            # table[i]=table[i].append(f,ignore_index=True)
                            table[i] = table[i].append(f)
                        else:
                            table[i]=f
                            c=1

                # if os.path.exists(dirName):
                    # for fn in os.listdir(dirName):
                    #     dirName1= os.path.join(dirName,fn)
                    #     v
                    #     # for z in range(len(Driver)):
                    #     #     print len(table[z])
                    #     f = pd.read_table(dirName1,sep=',')
                    #     if c!=0:
                    #         table[j]=table[j].append(f,ignore_index=True)
                    #     else:
                    #         table[j]=f
                    #         c=1
    # table = [TW,JH,HH,CS]
    # choose = [cTW,cJH,cHH,cCS]
    index = table[0].axes
    # pre_gauss = make_gauss(1, 0.8, 0.8)
    # ax = plt.figure().add_subplot(1, 1, 1)
    for i in range(1, 438):#drop ACU1::CF_Abg_DepInhEnt
        # ax = plt.figure().add_subplot(1, 1, 1)
        for j in range(0, 4):
            # try:
            if table[j].empty :
                print 'empty %s' %(Driver[j])
            else:
                choose[j] = table[j][index[1][i]]
                temp[j] = choose[j].dropna()  # remove Nan
                # print len(temp[j])
                m[j] = np.mean(temp[j])  # mean
                #print 'm'
                # print m[j]
                v[j] = np.var(temp[j])  # var
                # print 'v'
                # print v[j]
                temp[j] = temp[j].sort_values()
                gauss[j] = make_gauss(1, v[j], m[j])(temp[j])
                if len(temp[j]) != 0:
                    plt.plot(temp[j], gauss[j], color[j], linewidth=2)

                #######################################################
        # for j in range(0, 4):
        #     # try:
        #     if table[j].empty:
        #         print 'empty %s' % (Driver[j])
        #     else:
        #         choose[j] = table[j][index[1][125]]
        #         temp[j] = choose[j].dropna()  # remove Nan
        #         m[j] = np.mean(temp[j])  # mean
        #         v[j] = np.var(temp[j])  # var
        #         temp[j].sort_values()
        #         gauss[j] = make_gauss(1, v[j], m[j])(temp[j])
        #         if len(temp[j]) != 0:
        #             plt.plot(temp[j], gauss[j], color[j], linewidth=2)
        #         # ax.set_title(index[1][125])

        plt.title(index[1][i])
        plt.legend(['TW', 'JH', 'HH', 'CS'], loc='best')
        plt.grid()
        plt.axis('on')
        plt.savefig('/home/mskim/autron/graph/%s_%s.png' % (index[1][i], Datess[0]), dpi=128)
        # plt.clf()
# plt.plot(temp[0],gauss[0],color[0],temp[1],gauss[1],color[1],temp[2],gauss[2],color[2],temp[3],gauss[3],color[3])

    # ax=plt.figure().add_subplot(1,1,1)


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
    # choose = [cTW,cJH,cHH,cCS]
    choose = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    temp = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    main_index(Driver=Driver, Dates=Dates[i], Test=Test, color=color, table=table, choose=choose, m=m, v=v, gauss=gauss, temp=temp)


