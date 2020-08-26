import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df  = pd.read_csv('dot.csv')
# x_data = np.array(df[['x1','x2']])
x1 = np.array(df['x1']).reshape(-1,1)
x2 = np.array(df['x2']).reshape(-1,1)
x_data = np.hstack((x1,x2))

y_data = np.array(df["y_c"]).reshape(-1,1)

Y_c = [['red' if y else 'blue'] for y in y_data]

x_data = tf.cast(x_data, dtype=tf.float32)
y_data = tf.cast(y_data, dtype=tf.float32)

# print(x_data.numpy().shape)
# print(x_data[:,:,0])
# print(x_data[1,:])
"""
"""


train_db = tf.data.Dataset.from_tensor_slices((x_data,y_data)).batch(32)

# 建立参数
w1 = tf.Variable(tf.random.normal([2,11],stddev=0.1))
b1 = tf.Variable(tf.constant(0.01,shape=[11]))

w2 = tf.Variable(tf.random.normal([11,5],stddev=0.1))
b2 = tf.Variable(tf.constant(0.01,shape=[5]))

w3 = tf.Variable(tf.random.normal([5,1],stddev=0.1))
b3 = tf.Variable(tf.constant(0.01,shape=[1]))

lr = 0.005
epoch = 1000
# 训练
for epoch in range(epoch):
  for step,(x_train,y_train) in enumerate(train_db):
    with tf.GradientTape() as tape:
      h1 = tf.matmul(x_train,w1) +b1
      h1 = tf.nn.relu(h1)
      h2 = tf.matmul(h1, w2) + b2
      h2 = tf.nn.relu(h2)
      y = tf.matmul(h2,w3)+b3
      
      loss = tf.reduce_mean(tf.square(y-y_train))
    variables = [w1,b1,w2,b2,w3,b3]
    grads = tape.gradient(loss,variables)

    # 更新
    w1.assign_sub(lr*grads[0])
    b1.assign_sub(lr*grads[1])
    w2.assign_sub(lr*grads[2])
    b2.assign_sub(lr*grads[3])
    w3.assign_sub(lr * grads[4])
    b3.assign_sub(lr * grads[5])

  if epoch % 20 == 0:
    print('epoch:', epoch, 'loss:', float(loss))


#预测
print("---------predict------------")

xx,yy = np.mgrid[-3:3:1,-3:3:1]
grid = np.c_[xx.ravel(),yy.ravel()]
grid = tf.cast(grid,dtype=tf.float32)

probs = []
for x_test in grid:
  h1 = tf.matmul([x_test],w1)+b1
  h1 = tf.nn.relu(h1)
  h2 = tf.matmul(h1, w2) + b2
  h2 = tf.nn.relu(h2)
  y = tf.matmul(h2,w3) +b3
  probs.append(y)

x1 = x_data[:,0]
x2 = x_data[:,1]

# print(x_data)
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1,x2, color=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5])
plt.show()
"""
"""
