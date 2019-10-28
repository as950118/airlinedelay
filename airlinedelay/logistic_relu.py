'''
Maximum Accuracy : 98%
Mean Accuracy : 94%
'''

import tensorflow as tf
import numpy as np
import csv
from itertools import chain


data = np.loadtxt('./testdata.csv', delimiter=',', dtype = np.float32)
len_data = len(data[0])
arr = [[0],[0],[0],[0]]*len(data)
#for i in range(len(data)):#
#    arr[i] = data[i]
print(len_data)
Org_data = data[len_data//2:, :-1]
Org_dly = data[len_data//2:, [-1]]
Test_data = data[:len_data//2, :-1]
Test_dly = data[:len_data//2, [-1]]

x_data = Org_data
y_data = Org_dly
test_x_data = Test_data
test_y_data = Test_dly

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#Input Layer
W1 = tf.Variable(tf.random_uniform([len_data-1, len_data], -1.0, 1.0), name = 'weight1')
b1 = tf.Variable(tf.zeros([len_data]), name = 'bias1')
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

#Hidden Layer
#random_normal = 0~1사이의 균등확률 분포 값을 생성, 이경우엔 -1~1 사이
#zeros = n*n 영행렬 생성
W2 = tf.Variable(tf.random_uniform([len_data, len_data+1], -1.0, 1.0), name = 'weight2')
b2 = tf.Variable(tf.zeros([len_data+1]), name = 'bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
W3 = tf.Variable(tf.random_uniform([len_data+1, len_data+2], -1.0, 1.0), name = 'weight3')
b3 = tf.Variable(tf.zeros([len_data+2]), name = 'bias3')
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
W4 = tf.Variable(tf.random_uniform([len_data+2, len_data+3], -1.0, 1.0), name = 'weight4')
b4 = tf.Variable(tf.zeros([len_data+3]), name = 'bias4')
layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)
W5 = tf.Variable(tf.random_uniform([len_data+3, len_data+4], -1.0, 1.0), name = 'weight5')
b5 = tf.Variable(tf.zeros([len_data+4]), name = 'bias5')
layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)
W6 = tf.Variable(tf.random_uniform([len_data+4, len_data+5], -1.0, 1.0), name = 'weight6')
b6 = tf.Variable(tf.zeros([len_data+5]), name = 'bias6')
layer6 = tf.nn.relu(tf.matmul(layer5, W6) + b6)
W7 = tf.Variable(tf.random_uniform([len_data+5, len_data+6], -1.0, 1.0), name = 'weight7')
b7 = tf.Variable(tf.zeros([len_data+6]), name = 'bias7')
layer7 = tf.nn.relu(tf.matmul(layer6, W7) + b7)
W8 = tf.Variable(tf.random_uniform([len_data+6, len_data+7], -1.0, 1.0), name = 'weight8')
b8 = tf.Variable(tf.zeros([len_data+7]), name = 'bias8')
layer8 = tf.nn.relu(tf.matmul(layer7, W8) + b8)

#Output Layer
W9 = tf.Variable(tf.random_uniform([len_data+7, 1], -1.0, 1.0), name = 'weight9')
b9 = tf.Variable(tf.zeros([1]), name = 'bias9')

#출력결과가 0~1사이여야하므로 마지막단은 sigmoid를 사용함
hypothesis = tf.sigmoid(tf.matmul(layer8, W9) + b9)


#코스트 / 로쓰 함수
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

#정확도
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))


#실행
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(10001):
    sess.run(train, feed_dict = {X : x_data, Y : y_data})
    if step%2000 == 0:
        print(step,"\n", sess.run(cost, feed_dict = {X : x_data, Y : y_data}), "\n",sess.run([W1, W2, W3]))

h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {X : test_x_data, Y : test_y_data})
print("----------------\nHypothesis :\n", h[0:10], "\n----------------\nCorrect :\n", c[0:10], "\n----------------\nAccuracy : ", a);
h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {X : x_data, Y : y_data})
print("----------------\nHypothesis :\n", h[0:10], "\n----------------\nCorrect :\n", c[0:10], "\n----------------\nAccuracy : ", a);
