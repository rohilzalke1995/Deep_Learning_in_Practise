import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
dataset = pd.read_csv('/Users/rohilzalke/Desktop/ROHIL ZALKE/DataSet/Deep Learning A-Z/Part 4 - Self Organizing Maps (SOM)/Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

#Training The SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration=100)

#Visualizing The Results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som. distance_map().T)
colorbar() #The Highest Mean Interneuron Distance (MID) corresponde to the white color
#The patches which has MID low are very close to each other, also they are close to the wining nodes (MID=0) and hence forming a cluster.
#The patches with MID = 1, are outliers and fraud.

markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, 
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = None,
         markersize = 10,
         markeredgewidth = 2)
show()

#Finding The Frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)
