import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data

def data_check(data):
    print('Size of :')
    print('-Training-set "\t\t{}'.format(len(data.train.labels)))
    print('-Test-set "\t\t{}'.format(len(data.test.labels)))
    print('-Validation-set "\t\t{}'.format(len(data.validation.labels)))

data = input_data.read_data_sets("./data/MNIST/",one_hot=True)

data_test_cls = np.array([label.argmax() for label in data.test.labels])

print(len(data_test_cls), data_test_cls)

img_size = 28
img_flat = img_size*img_size
img_shape = (img_size,img_size)

num_classes = 10

with tf.graph():
    x = tf.placeholder([None,img_flat])
    y = tf.placeholder([None,num_classes])

    weight = tf.Variable(tf.zeros([img_flat,num_classes]))
    bias = tf.Variable(tf.zeros(num_classes))

    logit = tf.matmul(x,weight)+bias

    y_pred = tf.nn.softmax(logit)
    y_pred_cls = tf.argmax(y_pred)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=y)

    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(cost)


train_data = data.train
test_data = data.test