import numpy as np
from keras import layers, models, datasets
from keras.utils import np_utils
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN

import csv
reader = lambda rname: list(csv.reader(open(rname), delimiter=','))
writer = lambda wname: csv.writer(open(wname, 'w', newline=''))

class Data:
    def __init__(self, max_features=20000, maxlen=15):
        data = reader('testdata_2.csv')
        len_data = len(data)
        x_data = np.array([list(map(int, [e for e in elem[:15]])) for elem in data[:len_data]])
        y_data = np.array([int(elem[15]) for elem in data[:len_data]])
        X_data, y_data = ADASYN().fit_resample(x_data, y_data)
        y_data = np_utils.to_categorical(y_data)
        x_train, y_train, x_test, y_test = x_data[:len_data:2], y_data[:len_data:2], x_data[1:len_data:2], y_data[1:len_data:2]
        #x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        #x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test


class RNN_LSTM(models.Model):
    def __init__(self, max_features, maxlen):
        x = layers.Input((maxlen,))
        h = layers.Embedding(max_features, 128)(x)
        h = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(h)
        y = layers.Dense(2, activation='tanh')(h)
        super().__init__(x, y)

        self.compile(loss='binary_crossentropy',
                     optimizer='adam', metrics=['accuracy'])
class Machine:
    def __init__(self, max_features=20000, maxlen=15):
        self.data = Data(max_features, maxlen)
        self.model = RNN_LSTM(max_features, maxlen)

    def run(self, epochs=200, batch_size=32):
        data = self.data
        model = self.model

        print('Training stage')
        history = model.fit(data.x_train, data.y_train,
                  batch_size=batch_size, epochs=epochs,
                  validation_data=(data.x_test, data.y_test),
                  verbose=2)

        loss, acc = model.evaluate(data.x_test, data.y_test,
                                   batch_size=batch_size, verbose=2)

        print('Test performance: accuracy={0}, loss={1}'.format(acc, loss))
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

def main():
    m = Machine()
    m.run()

if __name__ == '__main__':
    main()