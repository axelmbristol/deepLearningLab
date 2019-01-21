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

names_ = [str(n) for n in range(1, 1087)]
names_.append("famacha_class")

data = pd.read_csv("C:\\Users\\fo18103\PycharmProjects\\famatchatable\\training_t_c.data", sep=",",
                   names=names_)
#
np.random.seed(0)
data = data.sample(frac=1).reset_index(drop=True)
# data = data.truncate(after=150)
print(data)
#
all_x = data[names_]
#
all_y = pd.get_dummies(data.famacha_class)
#

n_x = len(all_x.columns)
n_y = len(all_y.columns)

s_ = 450
train_x_s = np.split(all_x, [s_], axis=0)
train_y_s = np.split(all_y, [s_], axis=0)

train_x = train_x_s[0]
train_y = train_y_s[0]
test_x = train_x_s[1]
test_y = train_y_s[1]

print(test_x)
print(test_y)

print("n_x=%d, n_y=%d" % (n_x, n_y))
#3.2 Define a Perceptron!
x = tf.placeholder(tf.float32, shape=[None, n_x], name='input')
y = tf.placeholder(tf.float32, shape=[None, n_y], name='output')

# W = tf.Variable(tf.zeros([n_x, 3]), tf.float32, name='weight')
# b = tf.Variable(tf.zeros([1, 1]), tf.float32, name='bias')
# m = tf.add(tf.matmul(x, W), b)
# prediction = tf.nn.softmax(m)
# print(prediction)
#
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), axis=1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
#
# sess.run(tf.global_variables_initializer())
# accuracy = 0
# cpt = 0
# for i, epoch in enumerate(range(10000)):
#     sess.run([optimizer], feed_dict={x: train_x, y: train_y})
#     predictions = sess.run(prediction, feed_dict={x: test_x, y: test_y}).tolist()
#     #compute accuracy
#     for j in range(len(predictions)):
#         if prediction_is_right(predictions[j], test_y.values.tolist()[j]):
#             cpt += 1
#     accuracy = cpt/len(predictions)
#     cpt = 0
#
#     if(i % 100) == 0:
#         print("Accuracy of Perceptron at epoch %d is %.2f" % (epoch, accuracy))

#3.5 Define a DEEP fully connected network
layer1_s = 100
W_fc1 = tf.Variable(tf.truncated_normal([n_x, layer1_s], stddev=0.1), name="layer1_weights")
b_fc1 = tf.Variable(tf.constant(0.1, shape=[layer1_s]), name="layer1_bias")
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1, name="layer1_output")

layer2_s = 50
W_fc2 = tf.Variable(tf.truncated_normal([layer1_s, layer2_s], stddev=0.1), name="layer2_weights")
b_fc2 = tf.Variable(tf.constant(0.1, shape=[layer2_s]), name="layer2_bias")
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2, name="layer2_output")

layer3_s = 25
W_fc3 = tf.Variable(tf.truncated_normal([layer2_s, layer3_s], stddev=0.1), name="layer3_weight")
b_fc3 = tf.Variable(tf.constant(0.1, shape=[layer3_s]), name="layer3_bias")
h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3, name="layer3_output")

layer4_s = 10
W_fc4 = tf.Variable(tf.truncated_normal([layer3_s, layer4_s], stddev=0.1), name="layer4_weight")
b_fc4 = tf.Variable(tf.constant(0.1, shape=[layer4_s]), name="layer4_bias")
h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4, name="layer4_output")

layer5_s = 5
W_fc5 = tf.Variable(tf.truncated_normal([layer4_s, layer5_s], stddev=0.1), name="layer5_weight")
b_fc5 = tf.Variable(tf.constant(0.1, shape=[layer5_s]), name="layer5_bias")
h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5, name="layer5_output")

layer6_s = 3
W_fc6 = tf.Variable(tf.truncated_normal([layer5_s, layer6_s], stddev=0.1), name="layer6_weight")
b_fc6 = tf.Variable(tf.constant(0.1, shape=[layer6_s]), name="layer6_bias")
h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6, name="layer6_output")

layero_s = 2 #output class
W_fco = tf.Variable(tf.truncated_normal([layer6_s, layero_s], stddev=0.1), name="output_weight")
b_fco = tf.Variable(tf.constant(0.1, shape=[layero_s]), name="output_bias")
predictions_fcn = tf.nn.relu(tf.matmul(h_fc6, W_fco) + b_fco, name="predicted_output")

print(h_fc1, h_fc2, h_fc3, h_fc4, predictions_fcn)

n_w = n_x * layer1_s + \
      layer1_s * layer2_s +\
      layer2_s * layer3_s +\
      layer3_s * layer4_s +\
      layer4_s * layer5_s +\
      layer5_s * layer6_s +\
      layer6_s * layero_s
n_b = layer1_s + layer2_s + layer3_s + layer4_s + layer5_s + layer6_s + n_y
n = n_w + n_b
print("the network contains %d variables." % n)

with tf.name_scope('loss'):
    cost_fcn = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=predictions_fcn, scope="Cost_Function")
    tf.summary.scalar('loss', cost_fcn)

optimizer_fcn = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(cost_fcn)


accuracy = 0
cpt = 0
logs_path = "./logs/"
g = tf.get_default_graph()
with g.as_default():
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_path + '/train')
    test_writer = tf.summary.FileWriter(logs_path + '/test')
    sess.run(tf.global_variables_initializer())
    for i, epoch in enumerate(range(5000)):
        sess.run([optimizer_fcn], feed_dict={x: train_x, y: train_y})
        p = sess.run(predictions_fcn, feed_dict={x: test_x, y: test_y}).tolist()
        #compute accuracy
        for j in range(len(p)):
            if prediction_is_right(p[j], test_y.values.tolist()[j]):
                cpt += 1
        with tf.name_scope('accuracy'):
            accuracy = cpt/len(p)
            tf.summary.scalar('accuracy', accuracy)
        cpt = 0

        if(i % 100) == 0:
            print("Accuracy of my first dnn at epoch %d is %.2f" % (epoch, accuracy))

        summary_train, _ = sess.run([merged, optimizer_fcn], feed_dict={x: train_x, y: train_y})
        summary_test, _ = sess.run([merged, cost_fcn], feed_dict={x: test_x, y: test_y})

        train_writer.add_summary(summary_train, epoch)
        test_writer.add_summary(summary_test, epoch)