# Mathieu Blondel, September 2010
# License: BSD 3 clause

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import ExtractData_train_validation_svm as extractData
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import tensorflow as tf

num=1

# model parameter
TRAIN_SIZE =        1000
VALIDATION_SIZE =   1000

BATCH_SIZE = 100
NUM_EPOCHS = 5

NUM_ROWS =  300
DATA_SIZE = 3
NUM_CHANNELS = DATA_SIZE

DRIVERS = ['CS','HH','JH','TW']
# LABELS = '0,1,2,3'
LABELS = [0,1,2,3]

tf.app.flags.DEFINE_integer('num', num, "Set")
FLAGS = tf.app.flags.FLAGS
num = FLAGS.num
BASE = './sensor3_driver'
# train_data, train_labels = extractData.extract_data_oned(numRows=NUM_ROWS, numData=TRAIN_SIZE,
#                                                          drivers=DRIVERS, labels=LABELS,
#                                                          mode='train',
#                                                          DATA_SIZE=DATA_SIZE,
#                                                          NUM_CHANNELS=NUM_CHANNELS, ONED=True)
# train_data=[TRAIN_SIZE,600]


validation_data, validation_labels = extractData.extract_data_oned(numRows=NUM_ROWS,
                                                                   numData=VALIDATION_SIZE,
                                                                   drivers=DRIVERS, labels=LABELS,
                                                                   mode='validate',
                                                                   DATA_SIZE=DATA_SIZE,
                                                                   NUM_CHANNELS=NUM_CHANNELS,
                                                                   ONED=True, BASE=BASE)

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    print 'predictions'
    print np.shape(predictions)
    print 'labels'
    print np.shape(labels)
    count =0
    for i in range(len(predictions)-1):
        if predictions[i] != labels[i]:
            count+=1
    print count
    return (float(count)/float(VALIDATION_SIZE))*100




#Import Library
from sklearn import svm

model = svm.SVC(kernel='rbf', C=1, gamma=1)

for i in range(num):
    train_data, train_labels = extractData.extract_data_oned(numRows=NUM_ROWS, numData=TRAIN_SIZE,
                                                             drivers=DRIVERS, labels=LABELS,
                                                             mode='train',
                                                             DATA_SIZE=DATA_SIZE,
                                                             NUM_CHANNELS=NUM_CHANNELS, ONED=True, BASE=BASE)
    model.fit(train_data, train_labels)
    print ('train fit end')
    model.score(train_data, train_labels)
    print ('train score end')
#Predict Output
predicted= model.predict(validation_data)
print predicted

file=open('sensor3_svm.txt','a')
file.write(str(num*TRAIN_SIZE))
file.write(',')
file.write(str(error_rate(predicted, validation_labels)))
file.write('\n')
file.close()
print 'predicted error: %.1f%%' % error_rate(predicted, validation_labels)
print error_rate(predicted, validation_labels)
print TRAIN_SIZE

