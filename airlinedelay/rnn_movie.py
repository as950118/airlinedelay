from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import reuters

num_words = 2000
maxlen = 200

(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=num_words,test_split=0.2)
#print('X train',type(X_train), X_train, type(X_train[0]), X_train[0], type(X_train[0][0]), X_train[0][0])
#print('X test', X_test, type(X_test))
#print('Y train', type(Y_train), Y_train, type(Y_train[0]), Y_train[0], type(Y_train[0][0]), Y_train[0][0])
#print('Y test', Y_test, type(Y_test))

x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)
#print('x train',type(x_train), x_train, type(x_train[0]),x_train[0], type(x_train[0][0]),x_train[0][0])
#print('x test',x_test,type(x_test))
#print('y train',type(y_train), y_train, type(y_train[0]), y_train[0], type(y_train[0][0]),y_train[0][0])
#print('y test',y_test,type(y_test))
print(X_train)
print(x_train)
print(len(x_train),len(X_train))
print(len(x_train[0]),len(X_train[0]))
print(len(x_test), len(X_test))
print(len(x_test[0]), len(X_test[0]))
print(len(Y_train), len(y_train))
print(Y_train)
print(y_train)
'''
model = Sequential()
model.add(Embedding(num_words,maxlen))
model.add(LSTM(maxlen, activation='relu'))
model.add(Dense(len(y_test[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=maxlen, epochs=100, validation_data=(x_test, y_test))

print("\n 정확도 : %.4f" % (model.evaluate(x_test, y_test)[1]))

y_test_loss = history.history['val_loss']
y_train_loss = history.history['loss']

x_len = numpy.arange(len(y_test_loss))
plt.plot (x_len, y_test_loss, marker=',', c='red', label='Testset_loss')
plt.plot(x_len, y_train_loss, marker=',', c='blue', label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
'''