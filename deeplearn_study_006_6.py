import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1','x2']])
y_data = np.array(df['y_c']).reshape(-1,1)
Y_c = [['red' if y else 'blue'] for y in y_data]

# class DotModel(tf.keras.Model):
#   def __init__(self):
#     super(DotModel,self).__init__()
#     self.d1 = tf.keras.layers.Dense(11,activation='relu')
#     self.d2 = tf.keras.layers.Dense(2)
#
#   def call(self,x):
#     h1 = self.d1(x)
#     y = self.d2(h1)
#     return y

class DotModel(tf.keras.Model):
  def __init__(self):
    super(DotModel,self).__init__()
    self.d1 = tf.keras.layers.Dense(20,activation='relu')
    self.d2 = tf.keras.layers.Dense(2)

  def call(self,x):
    h1 = self.d1(x)
    y = self.d2(h1)
    return y

model = DotModel()
save_model_path = "./DotModel/Checkpoint/DotModel.ckpt"
model.load_weights(save_model_path)

#预测
print("---------predict------------")

xx,yy = np.mgrid[-3:3:1,-3:3:1]
grid = np.c_[xx.ravel(),yy.ravel()]


probs = model.predict(grid[tf.newaxis,...])
print(probs[:,:,1])
pred = tf.argmax(probs,axis=2)
print(pred)
pred = np.array(pred).reshape((-1,1))
tmp = np.hstack((grid,pred))
print(tmp)
# print(np.hstack((grid,)))
# print(grid)
x1 =x_data[:,0]
x2 =x_data[:,1]
probs = np.array(pred).reshape(xx.shape)
plt.scatter(x1,x2, color=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5])
plt.show()