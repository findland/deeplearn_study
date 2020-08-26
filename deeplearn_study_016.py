import tensorflow as tf
import os
import matplotlib.pyplot as plt

# 读取数据集
df = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test)=df.load_data()
x_train , x_test = x_train/255 , x_test/255

class AlexNetModel(tf.keras.Model):
  def __init__(self):
    super(AlexNetModel,self).__init__()
    self.c1 = tf.keras.layers.Conv2D(96,kernel_size=3)
    self.b1 = tf.keras.layers.BatchNormalization()
    self.a1 = tf.keras.layers.Activation(activation='relu')
    self.p1 = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2)

    self.c2 = tf.keras.layers.Conv2D(256, kernel_size=3)
    self.b2 = tf.keras.layers.BatchNormalization()
    self.a2 = tf.keras.layers.Activation(activation='relu')
    self.p2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)

    self.c3 = tf.keras.layers.Conv2D(384, kernel_size=3,padding="same",activation="relu")

    self.c4 = tf.keras.layers.Conv2D(384, kernel_size=3,padding="same",activation="relu")

    self.c5 = tf.keras.layers.Conv2D(256, kernel_size=3,padding="same",activation="relu")
    self.p5 = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2)

    self.flatten = tf.keras.layers.Flatten()
    self.f1 = tf.keras.layers.Dense(2048,activation='relu')
    self.fd1 = tf.keras.layers.Dropout(0.5)
    self.f2 = tf.keras.layers.Dense(2048,activation='relu')
    self.fd2 = tf.keras.layers.Dropout(0.5)
    self.f3 = tf.keras.layers.Dense(10,activation='softmax')



  def call(self,x):
    x = self.c1(x)
    x = self.b1(x)
    x = self.a1(x)
    x = self.p1(x)
    x = self.c2(x)
    x = self.b2(x)
    x = self.a2(x)
    x = self.p2(x)
    x = self.c3(x)
    x = self.c4(x)
    x = self.c5(x)
    x = self.p5(x)
    x = self.flatten(x)
    x = self.f1(x)
    x = self.fd1(x)
    x = self.f2(x)
    x = self.fd2(x)
    y = self.f3

    return y

model = AlexNetModel()
save_model_path = "./Cifar/AlexNet/checkpoint/AlexNet.ckpt"
if os.path.exists(save_model_path+'.index'):
  model.load_weights(save_model_path)
callback = tf.keras.callbacks.ModelCheckpoint(save_model_path,
                                              save_weights_only=True,
                                              save_best_only=True)

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
history = model.fit(x_train,y_train,batch_size=32,
                    validation_data=(x_test,y_test),validation_freq=1,
                    epochs=5,callbacks=[callback])

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
