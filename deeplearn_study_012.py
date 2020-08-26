import tensorflow as tf
import numpy as np
from PIL import Image
np.set_printoptions(threshold=np.inf)
# import matplotlib.pyplot as plt
path = "/home/lht/baidunetdiskdownload/class4/MNIST_FC/mnist_image_label/"
test_path = "/home/lht/baidunetdiskdownload/class4/MNIST_FC/"
model_save_path =  path +"checkpoint/" + "mnist.ckpt"
# model_save_path = './class4/checkpoint/mnist.ckpt'
class NumModel(tf.keras.Model):
  def __init__(self):
    super(NumModel,self).__init__()
    self.d1 = tf.keras.layers.Flatten()
    self.d2 = tf.keras.layers.Dense(128,activation="relu")
    self.d3 = tf.keras.layers.Dense(10,activation="softmax")
  def call(self,x):
    h1 = self.d1(x)
    h2 = self.d2(h1)
    y = self.d3(h2)
    return y
model = NumModel()
# model = tf.keras.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128,activation='relu'),
#   tf.keras.layers.Dense(10,activation='softmax')
# ])

model.load_weights(model_save_path)
'''
preNum =10
for i in range(preNum):
  image_path = input("the path of test picture:")
  img = Image.open(image_path)
  img = img.resize((28, 28), Image.ANTIALIAS)
  img_arr = np.array(img.convert('L'))

  img_arr = 255 - img_arr

  img_arr = img_arr / 255.0
  print("img_arr:", img_arr.shape)
  x_predict = img_arr[tf.newaxis, ...]
  print("x_predict:", x_predict.shape)
  result = model.predict(x_predict)

  pred = tf.argmax(result, axis=1)

  print('\n')
  tf.print(pred)

'''
imgs = []
for i in range(10):
  image_path = test_path+str(i)+".png"
  # print(image_path)
  img = Image.open(image_path)
  img = img.resize((28,28),Image.ANTIALIAS)
  img_arr = np.array(img.convert('L'))

  img_arr = 255 - img_arr

  img_arr = img_arr/255.0
  # result = model.predict(img_arr[tf.newaxis,...])
  # pred = tf.argmax(result, axis=1)
  # print("\n")
  # tf.print(pred)
  imgs.append(img_arr)
imgs = np.array(imgs)
# print(imgs)
# print(imgs.shape)
# x_predict = img_arr[tf.newaxis,...]
result = model.predict(imgs)
# print(result)
pred = tf.argmax(result,axis=1)
for i in range(10):
  print("\n")
  tf.print(pred[i])
