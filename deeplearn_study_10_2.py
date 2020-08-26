import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

mnist = tf.keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

class FashionModel(tf.keras.Model):
  def __init__(self):
    super(FashionModel,self).__init__()
    self.d1 = tf.keras.layers.Flatten()
    self.d2 = tf.keras.layers.Dense(20,activation='relu')
    self.d3 = tf.keras.layers.Dense(10,activation='relu')
    self.d4 = tf.keras.layers.Dense(10,activation='softmax')

  def call(self,x):
    h1 = self.d1(x)
    h2 = self.d2(h1)
    h3 = self.d3(h2)
    y =self.d4(h3)
    return y
model = FashionModel()

model_save_path = "./FashionModel/Checkpoint/fashion.ckpt"
if os.path.exists(model_save_path+'.index'):
  model.load_weights(model_save_path)
callback = tf.keras.callbacks.ModelCheckpoint(model_save_path,
                                              save_weights_only=True,
                                              save_best_only=True)

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy']
              )

history= model.fit(x_train,y_train,batch_size=32,epochs=100,
                   validation_freq=1,
                   validation_data=(x_test,y_test),
                   callbacks=[callback])

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
