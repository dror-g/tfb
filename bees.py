import numpy as np
import pandas as pd

bees = pd.read_csv('bees/train_labels.csv')
bees = bees.reindex(np.random.permutation(bees.index))

print len(bees), len(bees[bees.genus==1]), len(bees[bees.genus==0])

from scipy.ndimage import imread
from scipy.misc import imresize

bees['images'] = [imresize(imread('bees/images/train/'+str(bee)+'.jpg'),
                           (100, 100))[:, :, :3] for bee in bees.id]

from scipy.ndimage import imread
from scipy.misc import imresize

bees['images'] = [imresize(imread('bees/images/train/'+str(bee)+'.jpg'),
                           (100, 100))[:, :, :3] for bee in bees.id]

#import matplotlib.pyplot as plt
#%matplotlib inline
#
#plt.figure(figsize=(12, 4))
#for i in range(16):
#    plt.subplot(2, 8, i+1)
#    plt.imshow(bees.images[i+333])
#    plt.xticks([])
#    plt.yticks([])
#    plt.title(["Bombus", "Apis"][int(bees.genus[i+333])])
#plt.tight_layout()


import tensorflow as tf
sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 100, 100, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([25*25*64, 250])
b_fc1 = bias_variable([250])

h_pool2_flat = tf.reshape(h_pool2, [-1, 25*25*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([250, 2])
b_fc2 = bias_variable([2])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)




from random import sample, choice
from scipy.ndimage.interpolation import rotate

y_true = pd.get_dummies(bees.genus)

training_rows = range(0,3000)
test_rows = range(3000,3969)

X_test = np.concatenate([arr[np.newaxis] for arr in bees.images.loc[test_rows]/256.])

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1e-9))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())


#### Random rotated and train 
#for i in range(2501):   Epochs/iterations
for i in range(10):
    batch_rows = sample(training_rows, 50)
    X_training = np.concatenate([arr[np.newaxis] for arr in \
                                 bees.images.loc[batch_rows].apply(lambda x: rotate(x, choice([0,90,180,270]))/256.)])

    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: X_training,
                                                  y_: y_true.loc[batch_rows].values, 
                                                  keep_prob: 1.0})
        print "step {:d}, training accuracy {:.3f}".format(i, train_accuracy), 
        print "- test accuracy {:.3f}".format(accuracy.eval(feed_dict={x: X_test,
                                                                       y_: y_true.loc[test_rows].values,
                                                                       keep_prob: 1.0}))
    train_step.run(feed_dict={x: X_training,
                              y_: y_true.loc[batch_rows].values, 
                              keep_prob: 0.5})


### Standard training

for i in range(2555,10000):
    batch_rows = sample(training_rows, 50)
    X_training = np.concatenate([arr[np.newaxis] for arr in \
                                 bees.images.loc[batch_rows]\
                                         .apply(choice([(lambda x: x), np.fliplr, np.flipud]))\
                                         .apply(lambda x: rotate(x, choice([0,90,180,270]))/256.)])

    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: X_training,
                                                  y_: y_true.loc[batch_rows].values, 
                                                  keep_prob: 1.0})
        print "step {:d}, training accuracy {:.3f}".format(i, train_accuracy), 
        print "- test accuracy {:.3f}".format(accuracy.eval(feed_dict={x: X_test,
                                                                       y_: y_true.loc[test_rows].values,
                                                                       keep_prob: 1.0}))
    train_step.run(feed_dict={x: X_training,
                              y_: y_true.loc[batch_rows].values, 
                              keep_prob: 0.5})


