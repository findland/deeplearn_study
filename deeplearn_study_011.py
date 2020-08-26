from PIL import Image
import numpy as np
import os

np.set_printoptions(threshold=np.inf)

path = "/home/lht/baidunetdiskdownload/class4/MNIST_FC/mnist_image_label/"
train_path = "/home/lht/baidunetdiskdownload/class4/MNIST_FC/mnist_image_label/mnist_train_jpg_60000/"
test_path = "/home/lht/baidunetdiskdownload/class4/MNIST_FC/mnist_image_label/mnist_test_jpg_10000/"
train_txt ="/home/lht/baidunetdiskdownload/class4/MNIST_FC/mnist_image_label/mnist_train_jpg_60000.txt"
test_txt = "/home/lht/baidunetdiskdownload/class4/MNIST_FC/mnist_image_label/mnist_test_jpg_10000.txt"
x_train_savepath = "/home/lht/baidunetdiskdownload/class4/MNIST_FC/mnist_image_label/mnist_x_train.npy"
y_train_savepath ="/home/lht/baidunetdiskdownload/class4/MNIST_FC/mnist_image_label/mnist_y_train.npy"
x_test_savepath ="/home/lht/baidunetdiskdownload/class4/MNIST_FC/mnist_image_label/mnist_x_test.npy"
y_test_savepath ="/home/lht/baidunetdiskdownload/class4/MNIST_FC/mnist_image_label/mnist_y_test.npy"

def getdatabase (path,txt):
  x = []
  y = []
  n =0
  with open(txt) as filelist :
    for info in filelist.readlines():
      pic_file,feature = info.split()
      img = Image.open(path+pic_file)
      img = np.array(img.convert("L"))
      print("loading",pic_file)
      img = img/255.0
      x.append(img)
      y.append(feature)
      n+=1
  x = np.array(x)
  y = np.array(y)
  y = y.astype(np.int64)
  return x,y



# '''
import tensorflow as tf

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
    x_test_savepath) and os.path.exists(y_test_savepath) :
  # 没有记录时，读取数据集
  x_train_save = np.load(x_train_savepath)
  y_train = np.load(y_train_savepath)
  x_test_save = np.load(x_test_savepath)
  y_test = np.load(y_test_savepath)
  x_train = np.reshape(x_train_save,(len(x_train_save),28,28))
  x_test = np.reshape(x_test_save,(len(x_test_save),28,28))

else:

# 有记录时直接读取记录
  x_train , y_train =getdatabase(train_path,train_txt)
  x_test ,y_test = getdatabase(test_path,test_txt)

  x_train_save = np.reshape(x_train,(len(x_train),-1))
  x_test_save = np.reshape(x_test,(len(x_test),-1))

  np.save(x_train_savepath,x_train_save)
  np.save(y_train_savepath,y_train)
  np.save(x_test_savepath,x_test_save)
  np.save(y_test_savepath,y_test)



# mnist = tf.keras.datasets.mnist
# (x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train,x_test = x_train/255.0,x_test/255.0


# 建立模型
# model = tf.keras.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128,activation='relu'),
#   tf.keras.layers.Dense(10,activation='softmax')
# ])

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


# 配置优化方法
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 读取已存储的模型
checkpoint_save_path = path +"checkpoint/" + "mnist.ckpt"
if os.path.exists(checkpoint_save_path+".index"):
  print("-----------------load the model-----------------------")
  model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
print(cp_callback)

# 训练 优化
history = model.fit(x_train,y_train,batch_size=32 , epochs=200,
                    validation_data=(x_test,y_test),validation_freq=1,
                    callbacks=[cp_callback])
# 展示
model.summary()

# 将参数提取，存入文本
# print(model.trainable_variables)
with open(path+"weight.txt",'w') as file:
  for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape)+ '\n')
    file.write(str(v.numpy())+'\n')



# 显示 训练集 和 验证集 的acc 和loss 曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(acc,label="Training acc")
plt.plot(val_acc,label="Validation acc")
plt.title("Training and Validation Acc")
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label="Training loss")
plt.plot(val_loss,label="Validation loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()


# =======预测=============
test_path = "/home/lht/baidunetdiskdownload/class4/MNIST_FC/"


model.load_weights(checkpoint_save_path)
for i in range(10):
  image_path = test_path+str(i)+".png"
  print(image_path)
  img = Image.open(image_path)
  img = img.resize((28,28),Image.ANTIALIAS)
  img_arr = np.array(img.convert('L'))

  img_arr = 255.0 - img_arr

  img_arr = img_arr/255.0
  x_predict = img_arr[tf.newaxis,...]
  result = model.predict(x_predict)
  # print(result)
  pred = tf.argmax(result,axis=1)
  print(i,"\n")
  # print(pred)
  tf.print(pred)