import tensorflow as tf
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
# 在本例中采用了测试集
# 从训练集中划分的方法，
# 所以不在单独划分测试集
# validation_split=0.2

# 初始化模型
# models.Sequential搭建神经网络，
# tf.keras.layers.Dense(神经元个数，激活函数，正则化方法)
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(3,activation="softmax",kernel_regularizer=tf.keras.regularizers.l2())
])

# model.compile 配置训练方法
# 优化器 损失函数
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["sparse_categorical_accuracy"])

model.fit(x_train , y_train,batch_size=32,epochs=500,validation_split=0.2,validation_freq=20)

model.summary()
y= model.predict(x_train)
print(tf.argmax(y,axis=1))
print ("/////////////////////")
print(y_train)