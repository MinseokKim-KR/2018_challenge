import gzip
import os
import sys
import urllib

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# BASE = "../autron/driver_data_test"

BASE = "/home/mskim/autron/driver_data_170518"

i=0
c=0
# col = [i+row for row in range(0,438)]
# Driver = ['TW', 'JH', 'HH', 'CS']
# Dates = ['170220']
# Test = ['test1', 'test2', 'test3']
# color = ['b','r','y','g']

Driver = ['TW', 'JH', 'HH', 'CS']
# Driver = ['TW', 'JH', 'KT', 'CS']
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
#
# table = [TW,JH,HH,CS]
# choose = [cTW,cJH,cHH,cCS]
# table = [TW,JH,HH,CS]
table = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
choose = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
temp = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
m=[0,0,0,0]
v=[0,0,0,0]
gauss=[0,0,0,0]
#read all data and append table

def make_gauss(N, sig, mu):
    return lambda x: N/(sig*(2*np.pi)**.5) * np.e **(-(x-mu)**2/(2*sig**2))

# for i in range(len(Dates)):
#     for j in range(len(Driver)):
#         c=0

for j in range(len(Driver)):
    c=0
    for i in range(len(Dates)):
        for h in range(len(Test)):
            dirName = BASE+'/'+Dates[i]+'/'+Driver[j]+'/'+ Dates[i]+'_'+Test[h]+'_'+Driver[j]+'/'
            if os.path.exists(dirName):
                for fn in os.listdir(dirName):
                    dirName1= os.path.join(dirName,fn)
                    print dirName1
                    for z in range(len(Driver)):
                        print len(table[z])
                    f = pd.read_table(dirName1,sep=',')
                    if c!=0:
                        table[j]=table[j].append(f,ignore_index=True)
                    else:
                        table[j]=f
                        c=1
index = table[0].axes

# ax=plt.figure().add_subplot(1,1,1)

#
# #start drawing the picture
# for i in range(0,438):
#     ax = plt.figure().add_subplot(1, 1, 1)
#     for j in range(0,4):
#         choose[j] = table[j][index[1][i]]
#         choose[j]=table[j].dropna() #remove Nan
#         m[j] = numpy.mean(choose[j]) #mean
#         v[j] = numpy.var(choose[j]) #var
#         choose[j].sort()
#         gauss[j]=make_gauss(1,v[j], m[j])(choose[j])
#         ax.plot(choose[j], gauss[j], color[j], linewidth=2)
#         ax.set_title(index[j])
#     plt.legend(['TW', 'JH', 'HH', 'CS'], loc='best')
#     # plt.savefig()
#     plt.grid()
#     plt.axis('on')
#     plt.savefig('/home/mskim/autron/graph/%s_%s.png' % (index[i],Dates[0]), dpi=128)
#     # plt.close(ax)


for i in range(0,438):
    for j in range(0, 4):
        if len(table[j]) != 0:
            choose[j] = table[j][index[1][i]]
            choose[j] = choose[j].dropna()  # remove Nan
        # choose[j].sort_values()
        # mu, sigma = m[j], v[j]
        # x = mu + sigma * choose[j]
        # n, bins, patches = plt.hist(x, 50, normed=1, facecolor=color[j], alpha=0.75)
    plt.boxplot((choose[0], choose[1], choose[2], choose[3]))
    # plt.legend(['TW', 'JH', 'HH', 'CS'], loc='best')
    plt.axis('on')
    plt.grid(True)
    plt.savefig('/home/mskim/autron/p_clear_170518/%s_%s.png' % (index[1][i], Dates[0]), dpi=128)
    plt.clf()

# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# ax3 = fig.add_subplot(213)
# ax4 = fig.add_subplot(214)
#
# for j in range(0, 4):
#     if len(table[j]) != 0:
#         choose[j] = table[j][index[1][100]]
#         choose[j] = choose[j].dropna()  # remove Nan
#         m[j] = numpy.mean(choose[j])  # mean
#         v[j] = numpy.var(choose[j])  # var
#         choose[j].sort_values()
#         # mu, sigma = m[j], v[j]
#         # x = mu + sigma * choose[j]
#         # n, bins, patches = plt.hist(x, 50, normed=1, facecolor=color[j], alpha=0.75)
# plt.boxplot((choose[0],choose[1],choose[2],choose[3]))
# plt.legend(['TW', 'JH', 'HH', 'CS'], loc='best')
# plt.axis('on')
# plt.grid(True)
# plt.savefig('/home/mskim/autron/p_graph/%s_%s.png' % (index[1][100], Dates[0]), dpi=128)


#
# choose[0] = table[0][index[1][20]]
# choose[0] = choose[0].dropna()  # remove Nan
# m[0] = numpy.mean(choose[0])  # mean
# v[0] = numpy.var(choose[0])  # var
# choose[0].sort_values()
# mu, sigma = m[0], v[0]
# x = mu + sigma * choose[0]
# n, bins, patches = plt.hist(x, 50, normed=1, facecolor=color[0], alpha=0.75)
#
# choose[1] = table[0][index[1][21]]
# choose[1] = choose[1].dropna()  # remove Nan
# m[1] = numpy.mean(choose[1])  # mean
# v[1] = numpy.var(choose[1])  # var
# choose[j].sort_values()
# mu, sigma = m[1], v[1]
# x1 = mu + sigma * choose[1]
# n, bins, patches = plt.hist(x1, 50, normed=1, facecolor=color[1], alpha=0.75)
#
# choose[2] = table[0][index[1][22]]
# choose[2] = choose[2].dropna()  # remove Nan
# m[2] = numpy.mean(choose[2])  # mean
# v[2] = numpy.var(choose[2])  # var
# choose[2].sort_values()
# mu, sigma = m[2], v[2]
# x2= mu + sigma * choose[2]
# n, bins, patches = plt.hist(x2, 50, normed=1, facecolor=color[2], alpha=0.75)
#
# choose[3] = table[0][index[1][23]]
# choose[3] = choose[3].dropna()  # remove Nan
# m[3] = numpy.mean(choose[3])  # mean
# v[3] = numpy.var(choose[3])  # var
# choose[3].sort_values()
# mu, sigma = m[3], v[3]
# x3 = mu + sigma * choose[3]
# # plt.plot(x, 'r--', x1, 'bs', x2, 'g^', x3, 'cx' )
# n, bins, patches = plt.hist(x3, 50, normed=1, facecolor=color[3], alpha=0.75)
# # y = mlab.normpdf( bins, mu, sigma)
# # l = plt.plot(bins, y, 'r--', linewidth=1)
#
# choose[0] = table[0][index[1][20]]
# choose[0] = choose[0].dropna()  # remove Nan
# m[0] = numpy.mean(choose[0])  # mean
# v[0] = numpy.var(choose[0])  # var
# choose[0].sort_values()
# mu, sigma = m[0], v[0]
# x = mu + sigma * choose[0]
#
# choose[1] = table[0][index[1][20]]
# choose[1] = choose[1].dropna()  # remove Nan
# m[1] = numpy.mean(choose[1])  # mean
# v[1] = numpy.var(choose[1])  # var
# choose[j].sort_values()
# mu, sigma = m[1], v[1]
# x1 = mu + sigma * choose[1]
#
# choose[2] = table[0][index[1][20]]
# choose[2] = choose[2].dropna()  # remove Nan
# m[2] = numpy.mean(choose[2])  # mean
# v[2] = numpy.var(choose[2])  # var
# choose[2].sort_values()
# mu, sigma = m[2], v[2]
# x2= mu + sigma * choose[2]
#
# choose[3] = table[0][index[1][20]]
# choose[3] = choose[3].dropna()  # remove Nan
# m[3] = numpy.mean(choose[3])  # mean
# v[3] = numpy.var(choose[3])  # var
# choose[3].sort_values()
# mu, sigma = m[3], v[3]
# x3 = mu + sigma * choose[3]
# plt.boxplot((choose[0],choose[1],choose[2],choose[3]))

# plt.hist(x, 'r<', x1, 'bs', x2, 'g^', x3, 'cx' )









    # the histogram of the data
    # n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    # y = mlab.normpdf( bins, mu, sigma)
    # l = plt.plot(bins, y, 'r--', linewidth=1)

    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    # plt.legend(['TW', 'JH', 'HH', 'CS'], loc='best')
    # plt.axis('on')
    # plt.grid(True)
    # plt.savefig('/home/mskim/autron/graph/%s_%s.png' % (index[i], Dates[0]), dpi=128)
#
#
# def main_index(Driver= None, Dates=None, Test=None, color=None, table=None, choose=None, m=None, v=None, gauss=None, temp=None ):
#     Datess = [Dates]
#
#     for i in range(len(Driver)):
#         c=0
#         # print len(Driver)
#         for j in range(len(Datess)):
#             # print len(Datess)
#             for k in range(len(Test)):
#                 # print len(Test)
#                 # dirName = BASE+'/'+Datess[i]+'/'+Driver[j]+'/'+ Datess[i]+'_'+Test[h]+'_'+Driver[j]+'/'
#                 for l in range(len(Name)):
#                     # print len(Name)
#                     dirName = BASE + '/' + Datess[j] + '/' + Driver[i] + '/' + Datess[j] + '_' + Test[k] + '_' + Driver[i] + '/'+Datess[j] + '_' + Test[k] + '_' + Driver[i] + '_' + Name[l]+'.csv'
#                     if os.path.exists(dirName):
#                         print dirName
#                         f = pd.read_table(dirName, sep=',')
#                         if c != 0:
#                             # table[i]=table[i].append(f,ignore_index=True)
#                             table[i] = table[i].append(f)
#                         else:
#                             table[i]=f
#                             c=1
#
#     index = table[0].axes
#     for i in range(1, 438):
#
#         for j in range(0, 4):
#             # try:
#             if table[j].empty :
#                 print 'empty %s' %(Driver[j])
#             else:
#                 choose[j] = table[j][index[1][i]]
#                 temp[j] = choose[j].dropna()  # remove Nan
#                 # print len(temp[j])
#                 m[j] = np.mean(temp[j])  # mean
#                 #print 'm'
#                 # print m[j]
#                 v[j] = np.var(temp[j])  # var
#                 # print 'v'
#                 # print v[j]
#                 temp[j] = temp[j].sort_values()
#                 gauss[j] = make_gauss(1, v[j], m[j])(temp[j])
#                 if len(temp[j]) != 0:
#                     plt.plot(temp[j], gauss[j], color[j], linewidth=2)
#                 choose[j] = table[j][index[1][i]]
#                 choose[j] = table[j].dropna()  # remove Nan
#                 m[j] = numpy.mean(choose[j])  # mean
#                 v[j] = numpy.var(choose[j])  # var
#                 choose[j].sort()
#                 # gauss[j] = make_gauss(1, v[j], m[j])(choose[j])
#                 # ax.plot(choose[j], gauss[j], color[j], linewidth=2)
#                 # ax.set_title(index[j])
#                 mu, sigma = m[j], v[j]
#                 # x = mu + sigma*np.random.randn(10000)
#                 x = mu + sigma * choose[j]
#                 n, bins, patches = plt.hist(x, 50, normed=1, facecolor=color[j], alpha=0.75)
#
#         plt.title(index[1][i])
#         plt.legend(['TW', 'JH', 'HH', 'CS'], loc='best')
#         plt.grid()
#         plt.axis('on')
#         plt.savefig('/home/mskim/autron/graph/%s_%s.png' % (index[1][i], Datess[0]), dpi=128)
#
# for i in range(len(Dates)):
#
#     TW = pd.DataFrame()
#     JH = pd.DataFrame()
#     HH = pd.DataFrame()
#     CS = pd.DataFrame()
#
#     cTW = pd.DataFrame()
#     cJH = pd.DataFrame()
#     cHH = pd.DataFrame()
#     cCS = pd.DataFrame() #clear
#     table = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
#     # choose = [cTW,cJH,cHH,cCS]
#     choose = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
#     temp = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
#     main_index(Driver=Driver, Dates=Dates[i], Test=Test, color=color, table=table, choose=choose, m=m, v=v, gauss=gauss, temp=temp)
#
#
#
#
