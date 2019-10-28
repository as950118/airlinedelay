import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datas = pd.read_csv("AFSNT.csv", engine='python',encoding='cp949')
x_data = datas[['ARP_N', 'ODP_N']].as_matrix()
dly = datas[['DLY']].as_matrix()
y_data = [[1] if i=='N' else [0] for i in dly]
#x_data = np.array(x_data, np.float32)
y_data = np.array(y_data, np.float32)
print(x_data, type(x_data),y_data, type(y_data))
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2,1]), name='weight')#두개의 변수(X1,X2)가 나오고 Y로 출력됨 ==> 2,1
b = tf.Variable(tf.random_normal([1]), name='bias')#나가는값(Y)가 1개 ==> 1

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis>0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1001):
    cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
    if step % 200 == 0:
        print(step, cost_val)

h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
print("\nhypothesis : ", h, "\ncorrect : ", c, "\naccuracy : ", a);
