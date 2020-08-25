import tensorflow as tf
from sklearn.datasets import load_iris
import numpy as np
import os
import matplotlib.pyplot as plt

df = load_iris()
df = np.hstack((df.data,df.target.reshape(-1,1)))
np.random.seed(116)
np.random.shuffle(df)

x_train = df[:-30,:-1]
y_train = df[:-30,-1]
x_test = df[-30:,:-1]
y_test = df[-30:,-1]

class IrisModel(tf.keras.Model):
  def __init__(self):
    super(IrisModel,self).__init__()
    self.d1 = tf.keras.layers.Dense(12,activation="relu")
    self.d2 = tf.keras.layers.Dense(3,activation="softmax")

  def call(self,x):
    h1 = self.d1(x)
    y = self.d2(h1)
    return y
model = IrisModel()

save_model_path = "./IrisModel/Checkpoint/Iris.ckpt"
if os.path.exists(save_model_path+'.index'):
  model.load_weights(save_model_path)
callback = tf.keras.callbacks.ModelCheckpoint(save_model_path,
                                              save_best_only=True,
                                              save_weights_only=True)

model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
history = model.fit(x_train,y_train,batch_size=32,epochs=800,
                    callbacks=[callback],
                    validation_freq=1,
                    validation_data=(x_test,y_test))

model.summary()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1,2,1)
plt.plot(acc,label="Training acc")
plt.plot(val_acc,label="Testing acc")
plt.title("Training and Validation Acc")
plt.legend()
plt.subplot(1,2,2)
plt.plot(loss,label="Training loss")
plt.plot(val_loss,label="Testing loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()


