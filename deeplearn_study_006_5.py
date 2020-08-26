import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1','x2']])
y_data = np.array(df['y_c']).reshape(-1,1)
Y_c = [['red' if y else 'blue'] for y in y_data]

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
# 配置优化模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # loss = 'mse',
              metrics=['sparse_categorical_accuracy'])


save_model_path = "./DotModel/Checkpoint/DotModel.ckpt"

if os.path.exists(save_model_path+".index"):
  print("-----------------load model----------------------")
  model.load_weights(save_model_path)

callback = tf.keras.callbacks.ModelCheckpoint(save_model_path,
                                              save_weights_only=True,
                                              # save_best_only=True
                                              )

history = model.fit(x_data,y_data,batch_size=32,epochs=800,
                    # validation_data=(x_data,y_data),
                    # validation_freq=1,
                    callbacks=[callback])

model.summary()

acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']
plt.subplot(1,2,1)
plt.plot(acc,label="Training acc")
plt.title("Training and Validation Acc")
plt.legend()
plt.subplot(1,2,2)
plt.plot(loss,label="Training loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

'''
#预测
print("---------predict------------")

xx,yy = np.mgrid[-3:3:1,-3:3:1]
grid = np.c_[xx.ravel(),yy.ravel()]


probs = model.predict(grid[tf.newaxis,...])
print(probs[:,:,1])
pred = tf.argmax(probs,axis=2)
print(pred)
x1 =x_data[:,0]
x2 =x_data[:,1]
probs = np.array(pred).reshape(xx.shape)
plt.scatter(x1,x2, color=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5])
plt.show()
'''