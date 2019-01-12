############################################################
#                                                          #
#  Code for Lab 1: Your First Fully Connected Layer  #
#                                                          #
############################################################


import tensorflow as tf
import os
import os.path
import numpy as np
import pandas as pd


def prediction_is_right(prediction, ground_truth):
    i_g = ground_truth.index(1)
    i_p = prediction.index(max(prediction))
    return i_g == i_p


sess = tf.Session()

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", sep=",",
                   names=["sepal_length", "sepal_width", "petal_length", "petal_width", "iris_class"])
#

np.random.seed(0)
data = data.sample(frac=1).reset_index(drop=True)
#
all_x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
#
all_y = pd.get_dummies(data.iris_class)
#

n_x = len(all_x.columns)
n_y = len(all_y.columns)

train_x_s = np.split(all_x, [100], axis=0)
train_y_s = np.split(all_y, [100], axis=0)

train_x = train_x_s[0]
train_y = train_y_s[0]
test_x = train_x_s[1]
test_y = train_y_s[1]

print(test_x)
print(test_y)

#3.2 Define a Perceptron!
x = tf.placeholder(tf.float32, shape=[None, n_x], name='input')
y = tf.placeholder(tf.float32, shape=[None, n_y], name='output')


W = tf.Variable(tf.zeros([n_x, 3]), tf.float32, name='weight')
b = tf.Variable(tf.zeros([1, 1]), tf.float32, name='bias')

m = tf.add(tf.matmul(x, W), b)
prediction = tf.nn.softmax(m)
print(prediction)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

sess.run(tf.global_variables_initializer())
accuracy = 0
cpt = 0
for i, epoch in enumerate(range(10000)):
    sess.run([optimizer], feed_dict={x: train_x, y: train_y})
    var = sess.run(prediction, feed_dict={x: test_x, y: test_y}).tolist()
    #compute accuracy
    for j in range(len(var)):
        if prediction_is_right(var[j], test_y.values.tolist()[j]):
            cpt += 1
    accuracy = cpt/len(var)
    cpt = 0

    if(i % 100) == 0:
        print("Accuracy of Perceptron at epoch %d is %.2f" % (epoch, accuracy))

#3.5 Define a DEEP fully connected network
h1 = 10
W_fc1 = tf.Variable(tf.truncated_normal([n_x, h1], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[h1]))
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

h2 = 20
W_fc2 = tf.Variable(tf.truncated_normal([h1, h2], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[h2]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

h3 = 10
W_fc3 = tf.Variable(tf.truncated_normal([h2, h3], stddev=0.1))
b_fc3 = tf.Variable(tf.constant(0.1, shape=[h3]))
h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

h4 = 3
W_fc4 = tf.Variable(tf.truncated_normal([h3, h4], stddev=0.1))
b_fc4 = tf.Variable(tf.constant(0.1, shape=[h4]))
predictions_fcn = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

print(h_fc1, h_fc2, h_fc3, predictions_fcn)

n_w = n_x*10 + 10*20 + 20*10 + 10*3
n_b = 10 + 20 + 10 + n_y
n = n_w + n_b
print("the network contains %d variables." % n)

cost_fcn = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=predictions_fcn, scope="Cost_Function")
optimizer_fcn = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(cost_fcn)
sess.run(tf.global_variables_initializer())

accuracy = 0
cpt = 0
for i, epoch in enumerate(range(3001)):
    sess.run([optimizer_fcn], feed_dict={x: train_x, y: train_y})
    var = sess.run(predictions_fcn, feed_dict={x: test_x, y: test_y}).tolist()
    #compute accuracy
    for j in range(len(var)):
        if prediction_is_right(var[j], test_y.values.tolist()[j]):
            cpt += 1
    accuracy = cpt/len(var)
    cpt = 0

    if(i % 100) == 0:
        print("Accuracy of my first dnn at epoch %d is %.2f" % (epoch, accuracy))