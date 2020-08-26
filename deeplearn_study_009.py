import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train,y_train) ,(x_test,y_test) = mnist.load_data()
# 展示
# plt.imshow(x_train[0],cmap="gray")
# plt.show()
#
# print("x_train[0]\n",x_train[0])
# print("y_train[0]\n",y_train[0])
# print("x_train.shape\n",x_train.shape)
# print("y_train.shape\n",y_train.shape)

x_train ,x_test = x_train/255.0 , x_test/255.0
"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10,activation='softmax')
])
"""
class MinstModel(tf.keras.Model):
  def __init__(self):
    super(MinstModel,self).__init__()
    self.flat = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(128,activation='relu')
    self.d2 = tf.keras.layers.Dense(10,activation='softmax')
  def call(self,x):
    h1 = self.flat(x)
    h2 = self.d1(h1)
    y = self.d2(h2)
    return y

model = MinstModel()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1)
model.summary()
