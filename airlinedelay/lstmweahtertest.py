import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

reader = lambda rfname: pd.read_csv(rfname, delimiter=',', header=0, encoding='euc-kr')
np.random.seed(7)

rfname = 'merge_AFSNT_weather.csv'
usecols = [i for i in range(14,24)]
for usecol in usecols:
	try:
		dataframe = pd.read_csv(rfname, delimiter=',', usecols=[usecol], header=0, encoding='euc-kr')
		#dataset = dataframe['기온(°C)']
		#dataset = dataset.values
		dataset = dataframe.values
		#dataset = dataset[::]
		dataset = dataset.astype('float64')

		# scaling dataset
		scaler = MinMaxScaler(feature_range=(0,1))
		weather = scaler.fit_transform(dataset)

		# split dataset to train and test
		train_size = int(len(weather)*2//3)
		test_size =  len(weather) - train_size
		train, test = weather[0:train_size,:], weather[train_size:len(weather),:]

		# convert an array of values into a dataset matrix
		def create_dataset(dataset, look_back=1):
			dataX, dataY = [], []
			for i in range(len(dataset)-look_back-1):
				a = dataset[i:(i+look_back), 0]
				dataX.append(a)
				dataY.append(dataset[i + look_back, 0])
			return np.array(dataX), np.array(dataY)

		look_back = 1
		trainX, trainY = create_dataset(train, look_back)
		testX, testY = create_dataset(test, look_back)

		# reshape input to be [samples, time steps, features]
		trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
		testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
		#trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
		#testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

		# create and fit the LSTM network
		model = Sequential()
		model.add(LSTM(4, input_shape=(1, look_back)))
		#model.add(LSTM(4, input_shape=(look_back, 1), dropout=0.1))
		model.add(Dense(1))
		model.compile(loss='mean_squared_error', optimizer='adam')
		model.fit(trainX, trainY, epochs=100, validation_data=(testX, testY), batch_size=1, verbose=2, callbacks=[callbacks.EarlyStopping(patience=3, verbose=2)])
		#model.fit(trainX, trainY, epochs=30, validation_data=(testX, testY), batch_size=1, verbose=2)

		# make predictions
		trainPredict = model.predict(trainX)
		testPredict = model.predict(testX)

		# invert predictions
		trainPredict = scaler.inverse_transform(trainPredict)
		trainY = scaler.inverse_transform([trainY])
		testPredict = scaler.inverse_transform(testPredict)
		testY = scaler.inverse_transform([testY])

		# calculate root mean squared error
		trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
		print('Train Score: %.2f RMSE' % (trainScore))
		testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
		print('Test Score: %.2f RMSE' % (testScore))

		# shift train predictions for plotting
		trainPredictPlot = np.empty_like(dataset)
		trainPredictPlot[:, :] = np.nan
		trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

		# shift test predictions for plotting
		testPredictPlot = np.empty_like(dataset)
		testPredictPlot[:, :] = np.nan
		testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

		# plot baseline and predictions
		plt.plot(scaler.inverse_transform(dataset))
		plt.plot(trainPredictPlot)
		plt.plot(testPredictPlot)
		plt.title(usecol)
		i=0
		while 1:
			if os.path.isdir('./{0}/{1}'.format(usecol, str(i))):
				i += 1
			else:
				os.mkdir('./{0}/{1}'.format(usecol, str(i)))
				break
		plt.savefig('./{0}/{1}/test.png'.format(usecol, str(i)))
		plt.clf()
		plt.cla()
		plt.close()
	except Exception as e:
		print(e)
		continue