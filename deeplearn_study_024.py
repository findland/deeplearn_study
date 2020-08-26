import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


df = pd.read_csv("SH600519.csv")
data = df['open']

train_set = np.array(data[:len(data)-300]).reshape(-1,1)
test_set = np.array(data[len(data)-300:]).reshape(-1,1)

# 归一化
sc = MinMaxScaler()
train_set_scaler = sc.fit_transform(train_set)
test_set_scaler = sc.fit_transform(test_set)

# print(test_set_scaler)
x_train = []
y_train = []
x_test = []
y_test = []

for i in range(len(train_set)-60):
  x_train.append(train_set_scaler[i:i+60,0])
  y_train.append(train_set_scaler[i+60,0])

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
x_train,y_train = np.array(x_train),np.array(y_train)
x_train = x_train.reshape(len(x_train),60,1)


for i in range(len(test_set)-60):
  x_test.append(test_set_scaler[i:i+60,0])
  y_test.append(test_set_scaler[i+60,0])
x_test,y_test = np.array(x_test),np.array(y_test)
x_test = x_test.reshape(len(x_test),60,1)


model = tf.keras.Sequential([
  tf.keras.layers.LSTM(80,return_sequences=True),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.LSTM(100),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss = 'mean_squared_error')
checkpoint_save_path = "./Rnn/STOCK_LSTM/stock.ckpt"
if os.path.exists(checkpoint_save_path+'.index'):
  print('________loading model___________')
  model.load_weights(checkpoint_save_path)
callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_save_path,
                                              save_best_only=True,
                                              save_weights_only=True,
                                              monitor='val_loss')

history = model.fit(x_train,y_train,batch_size=64,epochs=50,
                    validation_data=(x_test,y_test),validation_freq=1,
                    callbacks=[callback])
model.summary()

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss,label="Training loss")
plt.plot(val_loss,label="Testing loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
pred_price = model.predict(x_test)
pred_price = sc.inverse_transform(pred_price)
real_price = sc.inverse_transform(test_set_scaler[60:])

plt.plot(real_price,color='red',label = "maotai")
plt.plot(pred_price,color='blue',label ='Pred')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(pred_price, real_price)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(pred_price, real_price))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求>均值）
mae = mean_absolute_error(pred_price, real_price)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)
