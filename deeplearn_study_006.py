import tensorflow as tf
import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1','x2']])
y_data = np.array(df['y_c'])
x_train = np.vstack(x_data)
y_train = np.vstack(y_data).reshape(-1,1)
Y_c = [['red' if y else 'blue'] for y in y_train]

# 转数据类型
x_train = tf.cast(x_train,dtype=tf.float32)
y_train = tf.cast(y_train,dtype=tf.float32)
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
# 2个隐藏层 11个神经元
w1 = tf.Variable(tf.random.normal([2,11]),dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01,shape=[11]))

w2 = tf.Variable(tf.random.normal([11,1]),dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01,shape=[1]))

lr = 0.005
epoch = 800

for epoch in range(epoch):
  for step , (x_train,y_train) in enumerate(train_db):
    with tf.GradientTape() as tape:
      h1 = tf.matmul(x_train,w1)+b1
      h1 = tf.nn.relu(h1)
      y = tf.matmul(h1,w2) + b2
      loss = tf.reduce_mean(tf.square(y-y_train))

      # 加入L2正则化
      # loss_regularization = []
      # loss_regularization.append(tf.nn.l2_loss(w1))
      # loss_regularization.append(tf.nn.l2_loss(w2))
      # loss_regularization = tf.reduce_sum(loss_regularization)
      # loss = loss + 0.03*loss_regularization

    variables = [w1,w2,b1,b2]
    grads = tape.gradient(loss,variables)

    #更新梯度
    w1.assign_sub(lr*grads[0])
    w2.assign_sub(lr*grads[1])
    b1.assign_sub(lr*grads[2])
    b2.assign_sub(lr*grads[3])

  if epoch % 20 ==0 :
    print('epoch:',epoch,'loss:',float(loss))

#预测
print("---------predict------------")

xx,yy = np.mgrid[-3:3:1,-3:3:1]
grid = np.c_[xx.ravel(),yy.ravel()]
grid = tf.cast(grid,dtype=tf.float32)

probs = []
for x_test in grid:
  h1 = tf.matmul([x_test],w1)+b1
  h1 = tf.nn.relu(h1)
  y = tf.matmul(h1,w2) +b2
  probs.append(y)

x1 = x_data[:,0]
x2 = x_data[:,1]

probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1,x2, color=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5])
plt.show()
