import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

punch_df=pd.read_csv('PUNCH.csv')
kick_df=pd.read_csv('KICK.csv')

x=[]
y=[]
timesteps=10

dataset = punch_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(timesteps, n_sample):
    x.append(dataset[i-timesteps:i,:])
    y.append(1)

dataset = kick_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(timesteps, n_sample):
    x.append(dataset[i-timesteps:i,:])
    y.append(0)

x, y = np.array(x), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model  = Sequential([
        LSTM(units = 50, return_sequences = True, input_shape = (x.shape[1], x.shape[2])),
        Dropout(0.2),
        LSTM(units = 50, return_sequences = True),
        Dropout(0.2),
        LSTM(units = 50, return_sequences = True),
        Dropout(0.2),
        LSTM(units = 50),
        Dropout(0.2),
        Dense(units = 1, activation="sigmoid"),
    ])
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")

model.fit(X_train, y_train, epochs=16, batch_size=32,validation_data=(X_test, y_test))
model.save("model.h5")
