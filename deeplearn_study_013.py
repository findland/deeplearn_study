import tensorflow as tf
import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd

df = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test) = df.load_data()
plt.imshow(x_train[0])
plt.show()
print("x_train[0]",x_train[0])
print("y_train[0]",y_train[0])
print("x_test.shape",x_test.shape)

# class CifarModel(tf.keras.Model):
#   def __init__(self):
#     super(CifarModel,self).__init__()
#     c1 = tf.
#     b1
#     a1
#     p1
#     d1