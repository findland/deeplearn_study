import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.fashion_mnist
(x_train,y_train) ,(x_test,y_test) = mnist.load_data()

'''
# 展示
plt.imshow(x_train[0],cmap="gray")
plt.show()

print("x_train[0]\n",x_train[0])
print("y_train[0]\n",y_train[0])
print("y_train\n",y_train)
print(type(y_test))
print("x_train.shape\n",x_train.shape)
print("y_train.shape\n",y_train.shape)


x_train,y_train = x_train/255 , y_train/255

# model = tf.keras.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128,activation='relu'),
#   tf.keras.layers.Dense(10,activation='softmax')
# ])
class FashionModel(tf.keras.Model):
  def __init__(self):
    super(FashionModel, self).__init__()
    self.d1 = tf.keras.layers.Flatten()
    self.d2 = tf.keras.layers.Dense(128,activation='relu')
    self.d3 = tf.keras.layers.Dense(64,activation='relu')
    self.d4 = tf.keras.layers.Dense(10,activation='softmax')
  def call(self,x):
    h1 = self.d1(x)
    h2 = self.d2(h1)
    h3 = self.d3(h2)
    y = self.d4(h3)
    return y
model = FashionModel()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train,y_train,batch_size=32 , epochs=5,validation_data=(x_test,y_test),validation_freq=1)
model.summary()

sum_num = y_test.shape
res = model.predict(x_test)
res = (np.argmax(res,axis=1))
print (type(res))
# res = tf.cast(res,dtype = tf.int32)
# y_test = tf.cast(y_test,dtype = tf.int32)
res = np.sum(np.equal(y_test, res))

print(sum_num,res)
'''
