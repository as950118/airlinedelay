from keras import layers, models, Sequential, callbacks
from keras.utils import np_utils
from keras.models import load_model
from keras_radam import RAdam

import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN

import pandas as pd
# reader = lambda rfname: pd.get_dummies(pd.read_csv(rfname, delimiter=',', header=0))
import csv

reader = lambda rfname: list(csv.reader(open(rfname), delimiter=','))
writer = lambda wfname: csv.writer(open(wfname, 'w', newline=''))

dirname='.'

class Data:
    def __init__(self, rfname, len_data=500000, len_col=5):
        data = reader(rfname)
        self.len_data = len_data
        self.len_col = len_col

        # feature(x_data)와 result(y_data) 나누기
        x_data = np.array([list(map(int, [elem[0], elem[1], elem[4]])) for elem in data[:self.len_data]])
        y_data = np.array([int(elem[self.len_col]) for elem in data[:self.len_data]])
        #print(x_data)
        #print(y_data)

        # 오버샘플링
        # 기온이므로 오버샘플링은 그닥 필요 없음
        # x_data, y_data = ADASYN(random_state=0).fit_resample(x_data, y_data)

        # result(기온) 원핫인코딩으로 카테고리화
        #y_data = np_utils.to_categorical(y_data)

        # 학습, 테스트 나누기
        x_train, y_train, = x_data[:len(x_data):2], y_data[:len(y_data):2],
        x_test, y_test = x_data[1:len(x_data):2], y_data[1:len(y_data):2]

        # 이니셜라이징
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test


class Model:
    def __init__(self, max_features=15000, output_embedding=128, dropout=0.1,
                 output_neuron=1, batch_size=2 ** 10, epochs=100):
        self.max_features = max_features

        self.output_embedding = output_embedding
        self.dropout = dropout
        self.output_neuron = output_neuron

        self.batch_size = batch_size
        self.epochs = epochs

        self.earlystop = callbacks.EarlyStopping(patience=3, verbose=2)

    def load_model(self):
        self.model = load_model(self.model_name)

    def save_model(self):
        self.model.save('{0}/{1}.h5'.format(dirname, self.model_name))

    def LSTM_Adam(self):
        self.model_name = "LSTM_Adam"
        model = Sequential()
        #model.add(layers.Embedding(self.max_features, self.output_embedding))
        model.add(layers.LSTM(3, input_shape=(1,3)))
        model.add(layers.Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        model.summary()

        self.model = model

    def LSTM_Adadelta(self):
        self.model_name = "LSTM_Adadelta"
        model = Sequential()
        model.add(layers.Embedding(self.max_features, self.output_embedding))
        model.add(layers.LSTM(self.output_embedding, dropout=self.dropout, recurrent_dropout=self.dropout))
        model.add(layers.Dense(self.output_neuron))
        model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])
        model.summary()

        self.model = model

    def LSTM_CNN(self):
        self.model_name = "LSTM_CNN"
        model = Sequential()
        model.add(layers.Embedding(self.max_features, self.output_embedding))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Conv1D(2 ** 8, 4, padding='valid', activation='relu', strides=1))
        model.add(layers.MaxPooling1D(pool_size=4))
        model.add(layers.LSTM(self.output_embedding))
        model.add(layers.Dense(self.output_neuron))
        model.add(layers.Activation('sigmoid'))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def LSTM_CNN_Adadelta(self):
        self.model_name = "LSTM_CNN_Adadelta"
        model = Sequential()
        model.add(layers.Embedding(self.max_features, self.output_embedding))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Conv1D(2 ** 8, 4, padding='valid', activation='relu', strides=1))
        model.add(layers.MaxPooling1D(pool_size=4))
        model.add(layers.LSTM(self.output_embedding))
        model.add(layers.Dense(self.output_neuron))
        model.add(layers.Activation('sigmoid'))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer='Adadelta', metrics=['accuracy'])

        self.model = model

    def LSTM_CNN_RAdam(self):
        self.model_name = "LSTM_CNN_RAdam"
        model = Sequential()
        model.add(layers.Embedding(self.max_features, self.output_embedding))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Conv1D(2 ** 8, 4, padding='valid', activation='relu', strides=1))
        model.add(layers.MaxPooling1D(pool_size=4))
        model.add(layers.LSTM(self.output_embedding))
        model.add(layers.Dense(self.output_neuron))
        model.add(layers.Activation('sigmoid'))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=RAdam(), metrics=['accuracy'])

        self.model = model


class Main():
    def __init__(self, rfname):
        self.data = Data(rfname)
        #self.model = Model(output_neuron=len(self.data.y_train[0]))
        self.model = Model(output_neuron=1)

    def train_model(self):
        self.data.x_train = np.reshape(self.data.x_train, (self.data.x_train.shape[0], 1, self.data.x_train.shape[1]))
        self.data.x_test = np.reshape(self.data.x_test, (self.data.x_test.shape[0], 1, self.data.x_test.shape[1]))
        print("----------------------------------")
        print("----------Training Start----------")
        print("----------------------------------")
        self.history = self.model.model.fit(self.data.x_train, self.data.y_train,
                                            batch_size=self.model.batch_size, epochs=self.model.epochs,
                                            validation_data=(self.data.x_test, self.data.y_test),
                                            verbose=2, callbacks=[self.model.earlystop])

    def test_model(self):
        loss, acc = self.model.model.evaluate(self.data.x_test, self.data.y_test,
                                              batch_size=self.model.batch_size, verbose=2)

        print('Test Results : acc={0} // loss={1}'.format(acc, loss))

    def loss_graph(self):
        y_train_loss = self.history.history['loss']
        y_test_loss = self.history.history['val_loss']

        x_len = np.arange(len(y_test_loss))

        plt.plot(x_len, y_test_loss, marker=',', c='red', label='Test loss')
        plt.plot(x_len, y_train_loss, marker=',', c='blue', label='Train loss')

        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(self.model.model_name)
        plt.savefig('{0}/{1}_loss.png'.format(dirname, self.model.model_name))
        plt.clf()
        plt.cla()
        plt.close()

    def acc_graph(self):
        y_train_acc = self.history.history['acc']
        y_test_acc = self.history.history['val_acc']

        x_len = np.arange(len(y_test_acc))

        plt.plot(x_len, y_test_acc, marker=',', c='pink', label='Test acc')
        plt.plot(x_len, y_train_acc, marker=',', c='green', label='Train acc')

        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title(self.model.model_name)
        plt.savefig('{0}/{1}_acc.png'.format(dirname, self.model.model_name))
        plt.clf()
        plt.cla()
        plt.close()

    def loss_acc_graph(self):
        y_train_loss = self.history.history['loss']
        y_test_loss = self.history.history['val_loss']
        y_train_acc = self.history.history['acc']
        y_test_acc = self.history.history['val_acc']

        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        loss_ax.plot(y_test_loss, c='red', label='Test loss')
        loss_ax.plot(y_train_loss, c='blue', label='Train loss')
        acc_ax.plot(y_test_acc, c='pink', label='Test acc')
        acc_ax.plot(y_train_acc, c='green', label='Train acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('acc')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')

        plt.title(self.model.model_name)
        plt.savefig('{0}/{1}_loss_acc.png'.format(dirname, self.model.model_name))
        plt.clf()
        plt.cla()
        plt.close()

    def run(self):
        self.model.LSTM_Adam()
        self.train_model()
        self.test_model()
        self.loss_graph()
        self.acc_graph()
        self.loss_acc_graph()
        self.model.save_model()

        self.model.LSTM_Adadelta()
        self.train_model()
        self.test_model()
        self.loss_graph()
        self.acc_graph()
        self.loss_acc_graph()
        self.model.save_model()

        self.model.LSTM_CNN()
        self.train_model()
        self.test_model()
        self.loss_graph()
        self.acc_graph()
        self.loss_acc_graph()
        self.model.save_model()

        self.model.LSTM_CNN_Adadelta()
        self.train_model()
        self.test_model()
        self.loss_graph()
        self.acc_graph()
        self.loss_acc_graph()
        self.model.save_model()

        self.model.LSTM_CNN_RAdam()
        self.train_model()
        self.test_model()
        self.loss_graph()
        self.acc_graph()
        self.loss_acc_graph()
        self.model.save_model()


if __name__ == "__main__":
    import os
    orgdirname = './test_weather/test'
    i = 1
    while 1:
        if os.path.isdir(orgdirname + str(i)):
            i += 1
        else:
            dirname = orgdirname + str(i)
            os.mkdir(dirname)
            print('Dirname : {0}'.format(dirname))
            Main("testdata_2.csv").run()
            break