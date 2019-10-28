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


def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data) - look_back):
        dataX.append(signal_data[i:(i + look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))


look_back = 40

# 1. 데이터셋 생성하기
#signal_data = np.cos(np.arange(1600) * (20 * np.pi / 1000))[:, None]

# 데이터 전처리
def preprocessing(signal_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    signal_data = scaler.fit_transform(signal_data)
    return signal_data

# 데이터 분리
def splitdata(signal_data):
    train = signal_data[len(signal_data)//2:]
    val = signal_data[len(signal_data)//2:(len(signal_data)*3)//4]
    test = signal_data[(len(signal_data)*3)//4:]
    return train, val, test

# 데이터셋 생성
def makedata(train,val,test,look_back=40):
    x_train, y_train = create_dataset(train, look_back)
    x_val, y_val = create_dataset(val, look_back)
    x_test, y_test = create_dataset(test, look_back)
    return x_train,y_train,x_val,y_val,x_test,y_test


# 데이터셋 전처리
def preprocessing_dataset(x_train,y_train,x_val,y_val,x_test,y_test):
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_train, x_val, x_test

# 2. 모델 구성하기
def modeling(look_back=40):
    model = Sequential()
    for i in range(2):
        model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True, return_sequences=True))
        model.add(Dropout(0.3))
    model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    return model

# 3. 모델 학습과정 설정하기
def modelcomplie(model):
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# 4. 모델 학습시키기
def modeltrain(model):
    custom_hist = CustomHistory()
    custom_hist.init()
    for i in range(200):
        model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, callbacks=[custom_hist],
                  validation_data=(x_val, y_val))
        model.reset_states()
    return model, custom_hist

# 5. 학습과정 살펴보기
def makegraph(custom_hist, usecol):
    plt.plot(custom_hist.train_loss)
    plt.plot(custom_hist.val_loss)
    plt.ylim(0.0, 0.15)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    i = 0
    while 1:
        if os.path.isdir('/{0}/{1}'.format(usecol, str(i))):
            i += 1
        else:
            os.mkdir('/{0}/{1}'.format(usecol, str(i)))
            break
        plt.savefig('./{0}/{1}/test.png'.format(usecol, str(i)))
    plt.clf()
    plt.cla()
    plt.close()

# 6. 모델 평가하기
def evalmodel(model, x_train,y_train, x_val,y_val, x_test,y_test):
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
def usemodel(model, x_test,y_test, usecol):
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
    i = 0
    while 1:
        if os.path.isdir('/{0}/{1}'.format(usecol, str(i))):
            i += 1
        else:
            os.mkdir('/{0}/{1}'.format(usecol, str(i)))
            break
        plt.savefig('./{0}/{1}/use.png'.format(usecol, str(i)))
    plt.clf()
    plt.cla()
    plt.close()


rfname = 'merge_AFSNT_weather.csv'
usecols = [i for i in range(14,24)]
for usecol in usecols:
    try:
        signal_data = pd.read_csv(rfname, delimiter=',', usecols=[usecol], header=0, encoding='euc-kr')
        signal_data = preprocessing(signal_data)
        train, val, test = splitdata(signal_data)
        x_train, y_train, x_val, y_val, x_test, y_test = makedata(train, val, test, look_back)
        x_train, x_val, x_test = preprocessing_dataset(x_train, y_train, x_val, y_val, x_test, y_test)
        model = modeling(look_back)
        model = modelcomplie(model)
        model, custom_hist = modeltrain(model)
        makegraph(custom_hist, usecol)
        trainScore, valScore, testScore = evalmodel(model, x_train, y_train, x_val, y_val, x_test, y_test)
        usemodel(model, x_test, y_test, usecol)
    except Exception as e:
        print(e)
        continue