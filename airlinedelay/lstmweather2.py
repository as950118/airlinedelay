# 0. 사용할 패키지 불러오기
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
reader = lambda rfname: pd.read_csv(rfname, delimiter=',', header=0, encoding='euc-kr')
np.random.seed(7)


class CustomHistory(keras.callbacks.Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

class lstmweahter:
    def __init__(self, rfname):
        self.look_back = 40
        self.rfname = rfname

    def create_dataset(self, signal_data, look_back=1):
        dataX, dataY = [], []
        for i in range(len(signal_data) - look_back):
            dataX.append(signal_data[i:(i + look_back), 0])
            dataY.append(signal_data[i + look_back, 0])
        return np.array(dataX), np.array(dataY)


    # 데이터 전처리
    def preprocessing(self, signal_data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        signal_data = scaler.fit_transform(signal_data)
        return signal_data

    # 데이터 분리
    def splitdata(self, signal_data):
        train = signal_data[len(signal_data)//2:]
        val = signal_data[len(signal_data)//2:(len(signal_data)*3)//4]
        test = signal_data[(len(signal_data)*3)//4:]
        return train, val, test

    # 데이터셋 생성
    def makedata(self, train,val,test,look_back=40):
        x_train, y_train = self.create_dataset(train, look_back)
        x_val, y_val = self.create_dataset(val, look_back)
        x_test, y_test = self.create_dataset(test, look_back)
        return x_train,y_train,x_val,y_val,x_test,y_test


    # 데이터셋 전처리
    def preprocessing_dataset(self, x_train,y_train,x_val,y_val,x_test,y_test):
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_train, x_val, x_test

    # 2. 모델 구성하기
    def modeling(self, look_back=40):
        model = Sequential()
        for i in range(2):
            model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True, return_sequences=True))
            model.add(Dropout(0.3))
        model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        return model

    # 3. 모델 학습과정 설정하기
    def modelcomplie(self, model):
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    # 4. 모델 학습시키기
    def modeltrain(self, model, x_train, y_train, x_val, y_val):
        custom_hist = CustomHistory()
        for i in range(5):
            model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, callbacks=[custom_hist],
                      validation_data=(x_val, y_val))
            model.reset_states()
        return model, custom_hist

    # 5. 학습과정 살펴보기
    def makegraph(self, custom_hist, usecol):
        plt.plot(custom_hist.train_loss)
        plt.plot(custom_hist.val_loss)
        plt.ylim(0.0, 0.15)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if not os.path.isdir('./weather/{0}'.format(usecol)):
           os.makedirs('./weather/{0}'.format(usecol))
        self.i = 0
        while 1:
            if os.path.exists('./weather/{0}/test_{1}_{2}.png'.format(usecol, str(self.i), self.point)):
                self.i += 1
            else:
                plt.savefig('./weather/{0}/test_{1}_{2}.png'.format(usecol, str(self.i), self.point))
                break
        plt.clf()
        plt.cla()
        plt.close()

    # 6. 모델 평가하기
    def evalmodel(self, model, x_train,y_train, x_val,y_val, x_test,y_test):
        trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
        model.reset_states()
        print('Train Score: ', trainScore)
        valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
        model.reset_states()
        print('Validataion Score: ', valScore)
        testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
        model.reset_states()
        print('Test Score: ', testScore)
        return trainScore, valScore, testScore

    # 7. 모델 사용하기
    def usemodel(self, model, x_test,y_test, usecol):
        look_ahead = 250
        xhat = x_test[0]
        predictions = np.zeros((look_ahead, 1))
        for i in range(look_ahead):
            prediction = model.predict(np.array([xhat]), batch_size=1)
            predictions[i] = prediction
            xhat = np.vstack([xhat[1:], prediction])

        plt.figure(figsize=(12, 5))
        plt.plot(np.arange(look_ahead), predictions, 'r', label="prediction")
        plt.plot(np.arange(look_ahead), y_test[:look_ahead], label="test function")
        plt.legend()
        #i = 0
        while 1:
            if os.path.exists('./weather/{0}/use_{1}_{2}.png'.format(usecol, str(self.i), self.point)):
                self.i += 1
            else:
                plt.savefig('./weather/{0}/use_{1}_{2}.png'.format(usecol, str(self.i), self.point))
                break
        plt.clf()
        plt.cla()
        plt.close()

    # 8. 모델저장
    def savemodel(self, model, trainScore, valScore, testScore, usecol):
        while 1:
            if os.path.exists('./weather/{0}/model_{1}_{2}.h5'.format(usecol, str(self.i), self.point)):
                self.i += 1
            else:
                model.save('./weather/{0}/model_{1}_{2}.h5'.format(usecol, str(self.i), self.point))
                break
        fw = open('./weather/{0}/score_{1}_{2}.txt'.format(usecol, str(self.i), self.point), 'w')
        fw.write('trainScore, valScore, testScore\n')
        fw.write('{0}, {1}, {2}'.format(str(trainScore), str(valScore), str(testScore)))
        fw.close()

    def run(self):
        rfname = self.rfname
        date_data = pd.read_csv(rfname, delimiter=',', usecols=[1], header=0, encoding='euc-kr')
        date_data_list = date_data.values.tolist()
        print(date_data)
        #date_point = [(idx, date_data_list[idx]) for idx in range(len(date_data_list)-1) if date_data_list[idx] != date_data_list[idx+1]]
        #date_point.append((len(date_data)-1, date_data[-1]))
        date_point = []
        pre = date_data_list[0]
        start = 0
        for idx, cur in enumerate(date_data_list):
            if pre != cur:
                date_point.append((start, idx, pre))
                start = idx
            pre = cur
        date_point.append( (start, len(date_data_list), pre) )
        print(date_point)
        usecols = [i for i in range(3,13)]
        for usecol in usecols:
            print(usecol)
            signal_dataset = pd.read_csv(rfname, delimiter=',', usecols=[usecol], header=0, encoding='euc-kr')
            signal_datas = [(signal_dataset[start:end], point) for start, end, point in date_point]
            try:
                for signal_data, point in signal_datas:
                    self.point = point
                    print("1. preprocessing")
                    signal_data = self.preprocessing(signal_data)
                    print("2. split data")
                    train, val, test = self.splitdata(signal_data)
                    print("3. make data")
                    x_train, y_train, x_val, y_val, x_test, y_test = self.makedata(train, val, test, self.look_back)
                    print("4. preprocessing data")
                    x_train, x_val, x_test = self.preprocessing_dataset(x_train, y_train, x_val, y_val, x_test, y_test)
                    print("5. modeling")
                    model = self.modeling(self.look_back)
                    print("6. model complie")
                    model = self.modelcomplie(model)
                    print("7. model train")
                    model, custom_hist = self.modeltrain(model, x_train, y_train, x_val, y_val)
                    print("8. make graph")
                    self.makegraph(custom_hist, usecol)
                    print("9. eval model")
                    trainScore, valScore, testScore = self.evalmodel(model, x_train, y_train, x_val, y_val, x_test, y_test)
                    print("10. use model")
                    self.usemodel(model, x_test, y_test, usecol)
                    print("11. save model")
                    self.savemodel(model, trainScore, valScore, testScore, usecol)
            except Exception as e:
                print(e)
                if not os.path.isdir('./weather/{0}'.format(usecol)):
                    os.makedirs('./weather/{0}'.format(usecol))
                i = 0
                while 1:
                    if os.path.exists('./weather/{0}/error_{1}_{2}.txt'.format(usecol, (i), self.point)):
                        i += 1
                    else:
                        fw = open('./weather/{0}/error_{1}_{2}.txt'.format(usecol, str(i), self.point), 'w')
                        fw.write(str(e))
                        fw.close()
                        break
                continue

if __name__== "__main__":
    lstmweahter('weather_2.csv').run()