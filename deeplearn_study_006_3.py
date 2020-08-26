import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dot.csv")
x_data = df[["x1","x2"]]
x_data = np.array(x_data)
y_data = np.array(df["y_c"]).reshape(-1,1)

Y_c = [['red' if y else 'blue'] for y in y_data]

x_data = tf.cast(x_data , dtype=tf.float32)
y_data = tf.cast(y_data , dtype=tf.float32)

x_data_square = tf.square(x_data)

train_db = tf.data.Dataset.from_tensor_slices((x_data_square,y_data)).batch(32)

epoch = 800
lr = 0.005

w1 = tf.Variable(tf.random.normal([2,1]))
b1 = tf.Variable(tf.constant(0.01,shape=[1]))

for epoch in range(epoch):
  for (x_train,y_train) in train_db:
    with tf.GradientTape() as tape:
      y = tf.matmul(x_train,w1) + b1
      y = tf.nn.relu(y)
      loss = tf.reduce_mean(tf.square(y-y_train))
    varitys = [w1,b1]
    grads = tape.gradient(loss,varitys)
    w1.assign_sub(lr*grads[0])
    b1.assign_sub(lr*grads[1])

  if epoch % 20 == 0:
    print('epoch:', epoch, 'loss:', float(loss))

#预测
print("---------predict------------")

xx,yy = np.mgrid[-3:3:1,-3:3:1]
# grid = np.c_[xx.ravel(),yy.ravel()]
# grid = tf.cast(grid,dtype=tf.float32)
grid_square  = np.c_[(xx*xx).ravel(),yy.ravel()]
grid_square = tf.cast(grid_square,dtype=tf.float32)

probs = []
for x_test in grid_square:
  y = tf.matmul([x_test],w1)+b1
  y = tf.nn.relu(y)
  probs.append(y)

x1 = x_data[:,0]
x2 = x_data[:,1]

# print(x_data)
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1,x2, color=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5])
plt.show()