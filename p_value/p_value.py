import gzip
import os
import sys
import urllib
#
# import tensorflow.python.platform
# from tensorflow.python.platform import gfile
import numpy
# import tensorflow as tf
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# BASE = "../autron/driver_data_test"

BASE = "/home/mskim/challenge_2018/data"
BASE = "../data"
i=0
c=0
# col = [i+row for row in range(0,438)]
# Driver = ['TW', 'JH', 'HH', 'CS']
# Dates = ['170220']
# Test = ['test1', 'test2', 'test3']
# color = ['b','r','y','g']

Driver = ['A','B','C','D','E','F','G','H','I']
# Driver = ['TW', 'JH', 'KT', 'CS']
# Dates = ['170217', '170220', '170518', '170530', '170601']
# Name = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11']
# Dates = ['170217']
# Test = ['test1', 'test2', 'test3']
color = ['b','r','y','g']


A=pd.DataFrame()
B=pd.DataFrame()
C=pd.DataFrame()
D=pd.DataFrame()
E=pd.DataFrame()
F=pd.DataFrame()
G=pd.DataFrame()
H=pd.DataFrame()
I=pd.DataFrame()



cA=pd.DataFrame()
cB=pd.DataFrame()
cC=pd.DataFrame()
cD=pd.DataFrame()
cE=pd.DataFrame()
cF=pd.DataFrame()
cG=pd.DataFrame()
cH=pd.DataFrame()
cI=pd.DataFrame()
# table = [TW,JH,HH,CS]
# choose = [cTW,cJH,cHH,cCS]
# table = [TW,JH,HH,CS]
table = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
choose = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
temp = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
m=[0,0,0,0,0,0,0,0,0]
v=[0,0,0,0,0,0,0,0,0]
gauss=[0,0,0,0,0,0,0,0,0]
#read all data and append table

def make_gauss(N, sig, mu):
    return lambda x: N/(sig*(2*np.pi)**.5) * np.e **(-(x-mu)**2/(2*sig**2))

# for i in range(len(Dates)):
#     for j in range(len(Driver)):
#         c=0

for j in range(len(Driver)):
    c=0
    dirName = BASE+'/'+Driver[j]+'/'
    if os.path.exists(dirName):
        for fn in os.listdir(dirName):
            dirName1= os.path.join(dirName,fn)
            print (dirName1)
            f = pd.read_table(dirName1,sep=',')
            if c!=0:
                table[j]=table[j].append(f,ignore_index=True)
            else:
                table[j]=f
                c=1
    for z in range(len(Driver)):
        print(len(table[z]))
index = table[0].axes


file=open('p_result.txt','a')
for i in range(0,53):
    # for j in range(0, 9):
        # if len(table[j]) != 0:
        #     choose[j] = table[j][index[1][i]]
        #     choose[j] = choose[j].dropna()  # remove Nan

    for j in range(0, 9):
        if len(table[j]) != 0:
            choose[j] = table[j][index[1][i]]
    f, p = stats.f_oneway(choose[0], choose[1], choose[2], choose[3],choose[4], choose[5], choose[6], choose[7], choose[8])
    print (choose[0])
    # plt.axis('on')
    # plt.grid(True)
    # plt.title(index[1][i])
    # plt.xlabel('number of drivers')
    # plt.savefig('../p_value/p_result/%s.png' % (index[1][i]), dpi=128)
    # plt.clf()

    plt.boxplot([choose[0], choose[1], choose[2], choose[3],choose[4], choose[5], choose[6], choose[7], choose[8]])
    # plt.legend(['TW', 'JH', 'HH', 'CS'], loc='best')
    plt.axis('on')
    plt.grid(True)
    plt.title(index[1][i])
    plt.xlabel('number of drivers')
    plt.savefig('../p_value/p_result/%s.png' % (index[1][i]), dpi=128)
    plt.clf()



    file.write(index[1][i])
    file.write(',')
    file.write(str(p))
    file.write('\n')
file.close()

    # plt.boxplot((choose[0], choose[1], choose[2], choose[3]))
    # # plt.legend(['TW', 'JH', 'HH', 'CS'], loc='best')
    # plt.axis('on')
    # plt.grid(True)
    # plt.title(index[1][i])
    # plt.xlabel('number of drivers')
    # plt.savefig('/home/mskim/autron/p_value/p_clear_18/%s_%s.png' % (index[1][i], Dates[0]), dpi=128)
    # plt.clf()
#
# file=open('p_result.txt','a')
# for i in range(0,438):
#     f,p = stats.f_oneway(table[0][index[1][i]], table[1][index[1][i]], table[2][index[1][i]], table[3][index[1][i]])
#     file.write(index[1][i])
#     file.write(',')
#     file.write(str(p))
#     file.write('\n')
# file.close()

# for j in range(0, 4):
#     if len(table[j]) != 0:
#         choose[j] = table[j][index[1][i]]
#         choose[j] = choose[j].dropna()  # remove Nan
#     # choose[j].sort_values()
#     # mu, sigma = m[j], v[j]
#     # x = mu + sigma * choose[j]
#     # n, bins, patches = plt.hist(x, 50, normed=1, facecolor=color[j], alpha=0.75)
# plt.boxplot((choose[0], choose[1], choose[2], choose[3]))
# # plt.legend(['TW', 'JH', 'HH', 'CS'], loc='best')
# plt.axis('on')
# plt.grid(True)
# plt.title(index[1][i])
# plt.xlabel('number of drivers')
# plt.savefig('/home/mskim/autron/p_value/p_clear_18/%s_%s.png' % (index[1][i], Dates[0]), dpi=128)
# plt.clf()
