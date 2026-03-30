#Stock Price Trend Prediction Using Recurrent Neural Network (RNN)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the training set

dataset_train = pd.read_csv('/Users/rohilzalke/Desktop/ROHIL ZALKE/DataSet/Deep Learning A-Z/Part 3 - Recurrent Neural Networks (RNN)/Google_Stock_Price_Train.csv')

training_set = dataset_train.iloc[:, 1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mn = MinMaxScaler(feature_range=(0,1))
training_set_scaled = mn.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []


for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Part-2: Building The RNN
#Importing the Keras Libraries and Packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

#Initialising The RNN
regressor = Sequential()

#Adding the first LSTM layer and some dropout regularisation
#Dropout regularisation is added to avoid overfitting
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding the second LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding the third LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding the forth LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Adding the Output Layer
regressor.add(Dense(units=1))

#Compiling The RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


#Part - 3: Making Prediction and Visualizing The Result

#Getting the stock price of 2017
dataset_test = pd.read_csv('/Users/rohilzalke/Desktop/ROHIL ZALKE/DataSet/Deep Learning A-Z/Part 3 - Recurrent Neural Networks (RNN)/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = mn.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = mn.inverse_transform(predicted_stock_price)

#Visualising The Results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#In the parts of the prediction containing some spikes, our predictions lag behind the actual values because our model cannot react to fast and non-linear changes.
#But for the parts predicting smooth changes, our model reacts very well. It also manages to follow the upward and downward trend.




