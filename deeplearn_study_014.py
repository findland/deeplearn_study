import tensorflow as tf
import matplotlib.pyplot as plt
import os
df = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test) = df.load_data()
# 将像素点归一化，
# 同时将数据类型
# 从int转化成float
x_train=x_train/255.0
x_test =x_test/255.0



class CifarModel(tf.keras.Model):
  def __init__(self):
    super(CifarModel,self).__init__()
    self.c1 = tf.keras.layers.Conv2D(filters=6,kernel_size=5,padding='same')
    self.b1 = tf.keras.layers.BatchNormalization()
    self.a1 = tf.keras.layers.Activation(activation="relu")
    self.p1 = tf.keras.layers.MaxPool2D()
    self.d1 = tf.keras.layers.Dropout(0.2)

    self.f1 = tf.keras.layers.Flatten()
    self.f2 = tf.keras.layers.Dense(128,activation='relu')
    self.d2 = tf.keras.layers.Dropout(0.2)
    self.fc = tf.keras.layers.Dense(10,activation="softmax")

  def call(self,x):
    x = self.c1(x)
    x = self.b1(x)
    x = self.a1(x)
    x = self.p1(x)
    x = self.d1(x)
    x = self.f1(x)
    x = self.f2(x)
    x = self.d2(x)
    y = self.fc(x)
    return y
model = CifarModel()

model_save_path = './Cifar/baseline/checkpoint/cifar.ckpt'
if os.path.exists(model_save_path+".index"):
  model.load_weights(model_save_path)
callback = tf.keras.callbacks.ModelCheckpoint(model_save_path,
                                              save_best_only=True,
                                              save_weights_only=True)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy']
              )

history = model.fit(x_train,y_train,batch_size=32,epochs=5,
                    callbacks= [callback],validation_freq=1,
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
