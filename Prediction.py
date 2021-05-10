import investpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


df = investpy.get_stock_historical_data(stock='BATA',
                                        country='india', 
                                        from_date='01/01/2010',
                                        to_date='29/11/2019') 


# LSTM 


dataset_train = df
train_set = dataset_train.iloc[:, 3:4].values

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(train_set)

X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 200, batch_size = 32)

df2 = investpy.get_stock_historical_data(stock='BATA',
                                        country='india',
                                        from_date='01/12/2019',
                                        to_date='31/12/2019'
                                       ) 
dataset_test = df2
real_stock_price = dataset_test.iloc[:, 3:4].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.figure(figsize=(20,10))
plt.plot(real_stock_price, color = 'black', label = 'BATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted BATA Stock Price')
plt.title('BATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('BATA Stock Price')
plt.legend()
plt.show()