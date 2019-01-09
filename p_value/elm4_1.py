from model import ELM
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import ExtractData_train_validation_elm as extractData_oned_train_val
import tensorflow as tf

num=1


# model parameter
TRAIN_SIZE =        1000
VALIDATION_SIZE =   1000

BATCH_SIZE = 100
NUM_EPOCHS = 5

NUM_ROWS =  300
DATA_SIZE = 4
NUM_CHANNELS = DATA_SIZE

DRIVERS = ['CS','HH','JH','TW']
# LABELS = '0,1,2,3'
LABELS = [0,1,2,3]


tf.app.flags.DEFINE_integer('num', num, "Set")
FLAGS = tf.app.flags.FLAGS
num = FLAGS.num


# Basic tf setting
tf.set_random_seed(2016)
sess = tf.Session()

# Get data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Construct ELM
batch_size = TRAIN_SIZE
hidden_num = 150
print("batch_size : {}".format(batch_size))
print("hidden_num : {}".format(hidden_num))
elm = ELM(sess, batch_size, NUM_ROWS*DATA_SIZE, hidden_num, 4)
BASE = './sensor4_1_driver'
# train_data, train_labels = extractData_oned_train_val.extract_data_oned(numRows=NUM_ROWS, numData=TRAIN_SIZE,
#                                                                         drivers=DRIVERS, labels=LABELS, mode='train',
#                                                                         DATA_SIZE=DATA_SIZE,
#                                                                         NUM_CHANNELS=NUM_CHANNELS, ONED=True)
# train_data=[TRAIN_SIZE,600]
validation_data, validation_labels = extractData_oned_train_val.extract_data_oned(numRows=NUM_ROWS,
                                                                                  numData=VALIDATION_SIZE,
                                                                                  drivers=DRIVERS, labels=LABELS,
                                                                                  mode='validate', DATA_SIZE=DATA_SIZE,
                                                                                  NUM_CHANNELS=NUM_CHANNELS, ONED=True, BASE=BASE)
# validation_data=[VALIDATION_SIZE,600]

# one-step feed-forward training
# train_x, train_y = mnist.train.next_batch(batch_size)
# elm.feed(train_data, train_labels)
# print('train end')


for i in range(num):
    train_data, train_labels = extractData_oned_train_val.extract_data_oned(numRows=NUM_ROWS, numData=TRAIN_SIZE,
                                                                            drivers=DRIVERS, labels=LABELS,
                                                                            mode='train',
                                                                            DATA_SIZE=DATA_SIZE,
                                                                            NUM_CHANNELS=NUM_CHANNELS, ONED=True, BASE=BASE)
    elm.feed(train_data, train_labels)
    print('train end')

# testing
elm.test(validation_data, validation_labels)
file=open('sensor4_1_elm.txt','a')
file.write(str(num*TRAIN_SIZE))
file.write(',')
file.write(str(elm.test(validation_data, validation_labels)))
file.write('\n')
file.close()
print TRAIN_SIZE
