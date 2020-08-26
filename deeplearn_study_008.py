import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn.datasets import load_iris
import numpy as np

# 划分数据集
df = load_iris()
# print(data.keys())
df = np.hstack((np.array(df.data),np.array(df.target.reshape(-1,1))))
np.random.seed(116)
np.random.shuffle(df)
x_train = df[:,:-1]
y_train = df[:,-1]

tf.random.set_seed(116)

class IrisModel(Model):
  def __init__(self):
    super(IrisModel,self).__init__()
    self.d1 = Dense(3,activation="sigmoid",kernel_regularizer=tf.keras.regularizers.l2())
  # 实现前向传播 x→y
  def call(self,x):
    y = self.d1(x)
    return y

model = IrisModel()

# model.compile 配置训练方法
# 优化器 损失函数
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["sparse_categorical_accuracy"])

model.fit(x_train , y_train,batch_size=32,epochs=500,validation_split=0.2,validation_freq=20)

model.summary()

y = model.predict(x_train)
print(y)