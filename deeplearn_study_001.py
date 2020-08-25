import tensorflow as tf

w = tf.Variable(tf.constant(5,dtype = tf.float16))
lr = 0.3 # 学习率
epoch = 40 # 迭代次数

for epoch in range(epoch): # 迭代次数
  with tf.GradientTape() as tape: # with 结构框起梯度下降计算过程
    loss = tf.square(w + 1)
  grads = tape.gradient(loss,w) # .gradient()告知函数对谁求导

  w.assign_sub(lr * grads) # 对变量自减 即： w -= lr*grads
  print("After %s epoch, w is %f, loss is %f"%(epoch,w,loss))

  # 在程序中w的损失函数是(w+1)^2，
  # 所以其最优值为w=-1
  # 不同的学习率会影响到其找到最优值的速度