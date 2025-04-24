import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import pandas_datareader as data 
import yfinance as yf 
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
import streamlit as st 



start = '2020-01-01'
end = '2025-03-31'

st.title('PredictoFin')

user_input = st.text_input("Enter Stock Ticker ", 'AAPL')
df = yf.download(user_input, start, end)


#Describing Data
st.subheader('Data from 2020 - 2025')
st.write(df.describe()) 


#Visualizations 
st.subheader('Closing Price VS Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price VS Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing Price VS Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)


data_training = pds.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pds.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Loading the LSTM keras Model 
model = load_model('Keras_model.h5', custom_objects={'Orthogonal': Orthogonal})

#testing part
past_100_days= data_training.tail(100)
final_df = pds.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test), np.array(y_test)
y_predict = model.predict(x_test)
scaler = scaler.scale_

my_list = [100]
scale_factor = 1/my_list[0]
y_predict = y_predict* scale_factor
y_test= y_test* scale_factor

#Final Grpah 
st.subheader('Predictiosn VS Original')
Fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'original price')
plt.plot(y_predict, 'r', label = 'predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(Fig2)
