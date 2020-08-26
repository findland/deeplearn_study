import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1','x2']])
y_data = np.array(df['y_c'])
x_train = np.vstack(x_data)
y_train = np.vstack(y_data).reshape(-1,1)
Y_c = [['red' if y else 'blue'] for y in y_train]
# print (x_train)
print(y_train)
# tf.random.set_seed()
# """

# class DotsModel(Model):
#   def __init__(self):
#     super(DotsModel,self).__init__()
#     self.d1 = Dense(11,activation="relu",kernel_regularizer=tf.keras.regularizers.l2())
#     self.d2 = Dense(2,activation='softmax')
#   # 实现前向传播 x→y
#   def call(self,x):
#     h1 = self.d1(x)
#     y = self.d2(h1)
#     return y

# model = tf.keras.Sequential([
#   tf.keras.layers.Dense(11,activation='relu'),
#   tf.keras.layers.Dense(2)
# ])

class DotModel(Model):
  def __init__(self):
    super(DotModel,self).__init__()
    self.d1 = tf.keras.layers.Dense(11,activation='relu')
    self.d2 = tf.keras.layers.Dense(5,activation='relu')
    self.d3 = tf.keras.layers.Dense(2,activation='softmax')

  def call(self,x):
    h1 = self.d1(x)
    h2 = self.d2(h1)
    y = self.d3(h2)
    return y
model=DotModel()

# model.compile 配置训练方法
# 优化器 损失函数
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              # loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              loss = tf.keras.losses.MeanSquaredError(),
              metrics="sparse_categorical_accuracy")

model.fit(x_train,y_train,batch_size=32,epochs=500,validation_split=0.2,validation_freq=20)

model.summary()


print("---------predict------------")

xx,yy = np.mgrid[-3:3:1,-3:3:1]
grid = np.c_[xx.ravel(),yy.ravel()]
# grid = tf.cast(grid,dtype=tf.float32)
# print(grid)
# print(model.predict([0,0]))
# print(model.predict(x_train))
probs = model.predict(grid)
print(probs)
probs =probs[:,1]
# '''
x1 = x_data[:,0]
x2 = x_data[:,1]

probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1,x2, color=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5])
plt.show()

# '''