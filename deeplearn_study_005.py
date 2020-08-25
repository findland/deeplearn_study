"""
常见损失函数
  均方差
  自定义
  交叉熵
    H(y_,y) = - sum (y_*ln(y))
    y_为真实概率
    y 为预测概率
    实现：
      tf.losses.categorical_crossentropy(y_,y)
    同时计算softmax和交叉熵的函数
      tf.nn.softmax_cross_entropy_with_logits(y_,y)

"""

import numpy as np
import tensorflow as tf
SEED = 23455
COST = 99
PROFIT = 1

rdm = np.random.RandomState(seed=SEED)
x  = rdm.rand(32,2)
y_ = [[x1+x2+(rdm.rand()/10-0.05)] for (x1,x2) in x] # 生成带噪声数据
x = tf.cast(x,dtype=tf.float32)

w1 = tf.Variable(tf.random.normal([2,1],stddev=1,seed=1))

epoch = 15000
lr =0.002

for epoch in range(epoch):
  with tf.GradientTape() as tape:
    y = tf.matmul(x,w1)
    # loss_mse = tf.reduce_mean(tf.square(y-y_))
    # loss_zdy = tf.reduce_sum(tf.where(np.greater(y_ , y),PROFIT*(y_-y),COST*(y-y_)))
    loss_zdy = tf.reduce_sum(tf.where(np.greater(y, y_), (y - y_)*COST,(y_ - y)*PROFIT ))
  grads = tape.gradient(loss_zdy,w1)
  w1.assign_sub(lr*grads)

  if epoch % 500 == 0:
    print("After %d training steps , w1 is "%(epoch))
    print(w1.numpy(),"\n")
print("Final w1 is :\n",w1.numpy())
