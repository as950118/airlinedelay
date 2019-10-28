import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)  # reproducibility

sample = ['1,1,1,3,6,1005,11.3,0,1.2,180,57,1028.3,1,0,6,10,0',
'1,1,1,3,6,930,7.7,0,2.2,180,68,1028.3,0.8,0,6,10,0',
'1,1,1,3,6,1245,12.6,0,1.7,360,57,1027.6,1,0,4,10,0',
'1,1,1,3,6,1325,12.6,0,2.3,360,57,1026.6,1,0,0,0,1',
'1,1,1,3,6,1610,11.7,0,2.9,50,64,1024.2,1,0,0,0,0',
'1,1,1,3,6,1645,11.7,0,2.9,50,64,1024.2,1,0,0,0,1',
'1,1,1,3,6,1930,11.1,0,0.7,200,68,1024,0,0,0,10,0',
'1,1,1,3,6,2035,10.6,0,1.9,180,70,1023.9,0,0,0,9,0',
'1,1,1,2,3,1705,9.8,0,1.7,290,47,1019.6,1,0,0,0,0',
'1,1,1,1,3,2025,4.2,0,1.9,70,73,1015.4,0,0,0,11,0',
'1,1,1,1,3,1240,4,0,1.3,50,69,1018.7,0.2,0,9,8,0',
'1,1,1,1,3,1325,5.1,0,1.4,20,65,1017.9,0.1,0,8,8,0',
'1,1,1,1,3,805,-1.3,0,1.4,20,87,1019,0,0,8,10,0',
'1,1,1,4,3,1745,8.5,0,1.6,340,48,1019.9,0.8,0,4,0,0',
'1,1,1,4,3,1825,7.8,0,1.2,290,52,1019.6,0,0,4,0,0',
'1,1,1,1,3,905,-0.4,0,1.6,20,83,1019.4,0,0,9,10,0',
'1,1,1,2,3,2105,5.4,0,0.7,360,66,1018.7,0,0,0,0,0',
'1,1,1,1,3,1855,5.6,0,2.4,50,70,1015.8,0,0,8,9,1',
'1,1,1,1,3,1130,2.5,0,1.9,50,71,1019.9,0.6,0,9,10,0',
'1,1,1,1,3,1315,5.1,0,1.4,20,65,1017.9,0.1,0,8,8,0',
'1,1,1,1,3,2100,4,0,2.6,50,74,1014.6,0,0,9,10,0',
'1,1,1,8,3,1545,10.5,0,1.2,290,56,1018.8,1,0,0,0,0',
'1,1,1,8,3,1620,10.1,0,0.3,0,55,1018.2,1,0,0,0,1',
'1,1,1,4,3,930,0.4,0,1.3,340,88,1023.8,0,0,6,0,0',
'1,1,1,14,3,1215,1.7,0,0,0,74,1011.7,0,0,0,0,0',
'1,1,1,14,3,1310,3.3,0,0.5,320,68,1010.4,0.2,0,0,0,0',
'1,1,1,12,3,1615,10.4,0,1.3,160,37,1023.1,1,0,0,0,0',
'1,1,1,1,3,1425,6.7,0,0.7,340,61,1017.1,0.7,0,8,7,0',
'1,1,1,1,3,815,-1.3,0,1.4,20,87,1019,0,0,8,10,0',
'1,1,1,12,3,1710,8,0,0.5,180,48,1022.7,0.7,0,0,0,0',
'1,1,1,1,3,1910,4.8,0,2.2,50,73,1015.5,0,0,0,11,0',
'1,1,1,6,3,1110,2.1,0,0.9,340,82,1023.9,0.5,0,8,10,0',
'1,1,1,6,3,830,-0.2,0,0.7,180,90,1023.1,0,0,9,10,0',
'1,1,1,6,3,1145,2.1,0,0.9,340,82,1023.9,0.5,0,8,10,0',
'1,1,1,6,3,1430,7.8,0,1.5,180,56,1020.8,0.9,0,5,0,0',
'1,1,1,1,3,2200,3.9,0,1.4,20,75,1014.4,0,0,0,10,1',
'1,1,1,8,3,920,1.5,0,0.3,0,94,1022.4,0.3,0,8,10,0',
'1,1,1,8,3,1000,2.8,0,0.7,250,87,1022.7,0.2,0,9,10,0',
'1,1,1,8,3,1230,6.7,0,1,230,68,1021.5,1,0,0,0,0',
'1,1,1,1,3,645,-1.5,0,1.6,20,87,1018.1,0,0,3,7,0',
'1,1,1,1,3,1155,2.5,0,1.9,50,71,1019.9,0.6,0,9,10,0',
'1,1,1,1,3,1950,4.8,0,2.2,50,73,1015.5,0,0,0,11,0',
'1,1,1,6,3,1510,8.4,0,1.1,250,53,1020.2,0.9,0,5,10,1',
'1,1,1,1,3,1825,5.6,0,2.4,50,70,1015.8,0,0,8,9,0',
'1,1,1,1,3,2010,4.2,0,1.9,70,73,1015.4,0,0,0,11,0',
'1,1,1,4,3,1050,1.9,0,0.3,0,83,1024.1,0,0,7,0,0',
'1,1,1,4,3,1130,4.9,0,1.9,290,69,1023.9,0.4,0,7,0,0',
'1,1,1,8,3,1310,9.9,0,1.3,250,57,1020.3,1,0,0,0,0',
'1,1,1,5,3,1550,10.8,0,1.9,50,44,1015.9,0.9,0,0,0,0',
'1,1,1,5,3,1630,10.4,0,1.7,70,44,1015.8,0.5,0,2,0,0',
'1,1,1,9,3,1900,7.5,0,1.8,290,62,1017.8,0,0,0,0,0',
'1,1,1,1,3,1800,5.6,0,2.4,50,70,1015.8,0,0,8,9,0',
'1,1,1,1,3,1230,4,0,1.3,50,69,1018.7,0.2,0,9,8,0',
'1,1,1,9,3,915,3.6,0,1.5,230,72,1022.5,0.7,0,2,0,0',
'1,1,1,1,3,2045,4.2,0,1.9,70,73,1015.4,0,0,0,11,0',
'1,1,1,1,3,1610,6.8,0,1.2,230,67,1016.5,0,0,9,7,0',
'1,1,1,13,3,1400,10.3,0,2.1,180,65,1025.3,1,0,0,0,0',
'1,1,1,13,3,1435,10.3,0,2.1,180,65,1025.3,1,0,0,0,0',
'1,1,1,8,3,1645,10.1,0,0.3,0,55,1018.2,1,0,0,0,0',
'1,1,1,8,3,1730,9.7,0,0.3,0,57,1017.6,1,0,0,0,0',
'1,1,1,2,3,805,1.5,0,1,340,80,1023,0,0,0,0,0']
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex
ret_sample = '1,1,1,1,3,0905,00.4,0,1.6,020,83,1019.4,0,0,9,10,1'
ret_idx2char = list(set(ret_sample))  # index -> char
ret_char2idx = {c: i for i, c in enumerate(ret_idx2char)}  # char -> idex

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello
ret_sample_idx = [ret_char2idx[c] for c in ret_sample]
ret_x_data = [ret_sample_idx[:-1]]

X = tf.compat.v1.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.compat.v1.placeholder(tf.int32, [None, sequence_length])  # Y label

x_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
#cell = tf.contrib.rnn.BasicLSTMCell(
#    num_units=hidden_size, state_is_tuple=True)
cell = tf.compat.v1.nn.rnn_cell.LSTMCell(
    num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: ret_x_data})

        # print char using dic
        result_str = [ret_idx2char[c] for c in np.squeeze(result)]

        print(i, "loss:", l, "Prediction:", ''.join(result_str))