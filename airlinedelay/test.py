from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import csv
reader = lambda rname:list(csv.reader(open(rname), delimiter=','))
writer = lambda wname:csv.writer(open(wname, 'w', newline=''))
data = reader('testdata_2.csv')
'''
#data = list([elem.split(',') for elem in sample])
#y_data = np.array([list(map(float, [e for e in elem[15]])) for elem in data[:2000]])
#x_data = np.array([list(map(int, [e for e in elem[:6]])) for elem in data[:9000]])

len_data = 10000
x_data = np.array([list(map(int, [elem[0], elem[1], elem[4]])) for elem in data[:len_data]])
y_data = np.array([int(elem[5]) for elem in data[:len_data]])
len_elem = len(x_data[0])
y_data = np_utils.to_categorical(y_data)
x_train, y_train, x_test, y_test = x_data[:len_data:2], y_data[:len_data:2], x_data[1:len_data:2], y_data[1:len_data:2]
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
len_ret = len(y_train[0])
print(len_ret)
'''
'''
x_train = sequence.pad_sequences(x_train)
x_test = sequence.pad_sequences(x_test)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
'''
'''
batch_size = 10300
model = Sequential()
model.add(Embedding(batch_size, len_elem))
model.add(LSTM(batch_size, activation='tanh'))
model.add(Dense(len_ret, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=200, validation_data=(x_test, y_test))
print("Accuracy : %.4f" %(model.evaluate(x_test, y_test)[1]))

y_train_loss = history.history['loss']
y_test_loss = history.history['val_loss']

x_len = np.arange(len(y_test_loss))

plt.plot (x_len, y_test_loss, marker=',', c='red', label='Testset_loss')
plt.plot(x_len, y_train_loss, marker=',', c='blue', label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
'''
import datetime
