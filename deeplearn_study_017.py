import tensorflow as tf
import os
import matplotlib.pyplot as plt

# 读取数据集
df = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test)=df.load_data()
x_train , x_test = x_train/255 , x_test/255

# 通用CBA层
class ConvBNRelu(tf.keras.Model):
  def __init__(self,ch,kernelsz=3,strides =1,padding="same"):
    super(ConvBNRelu,self).__init__()
    self.model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(ch,kernelsz,strides=strides,padding=padding),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation(activation='relu')
    ])
  def call(self,x):
    x = self.model(x)
    return x

# Inception 模块
class InceptionBlk(tf.keras.Model):
  def __init__(self,ch,strides=1):
    super(InceptionBlk,self).__init__()
    # self.c1 = tf.keras.layers.Conv2D(16,kernel_size=1,padding='same')
    # self.b1 = tf.keras.layers.BatchNormalization()
    # self.a1 = tf.keras.layers.Activation(activation='relu')
    self.c1 = ConvBNRelu(ch,kernelsz=1,strides=strides)

    # self.c21 = tf.keras.layers.Conv2D(16,kernel_size=1,padding='same')
    # self.b21 = tf.keras.layers.BatchNormalization()
    # self.a21 = tf.keras.layers.Activation(activation='relu')
    self.c21 = ConvBNRelu(ch,kernelsz=1,strides=strides)

    # self.c22 = tf.keras.layers.Conv2D(16,kernel_size=3,padding='same')
    # self.b22 = tf.keras.layers.BatchNormalization()
    # self.a22 = tf.keras.layers.Activation(activation='relu')
    self.c22 = ConvBNRelu(ch,kernelsz=3,strides=1)

    # self.c31 = tf.keras.layers.Conv2D(16,kernel_size=1,padding='same')
    # self.b31 = tf.keras.layers.BatchNormalization()
    # self.a31 = tf.keras.layers.Activation(activation='relu')
    self.c31 = ConvBNRelu(ch,kernelsz=3,strides=strides)

    # self.c32 = tf.keras.layers.Conv2D(16,kernel_size=5,padding='same')
    # self.b32 = tf.keras.layers.BatchNormalization()
    # self.a32 = tf.keras.layers.Activation(activation='relu')
    self.c32 = ConvBNRelu(ch,kernelsz=5,strides=1)

    self.p41 = tf.keras.layers.MaxPool2D(pool_size=3,strides=1,padding='same')
    # self.c42 = tf.keras.layers.Conv2D(16,kernel_size=1,padding='same')
    # self.b42 = tf.keras.layers.BatchNormalization()
    # self.a42 = tf.keras.layers.Activation(activation='relu')
    self.c42 = ConvBNRelu(ch,kernelsz=1,strides=strides)




  def call(self,x):
    h1 = self.c1(x)

    h2 = self.c21(x)
    h2 = self.c22(h2)

    h3 = self.c31(x)
    h3 = self.c32(h3)

    h4 = self.p41(x)
    h4 = self.c42(h4)

    x = tf.concat([h1,h2,h3,h4],axis=3)

    return x

class Inception10(tf.keras.Model):
  def __init__(self,num_blocks,num_classes,init_ch=16,**kwargs):
    super(Inception10,self).__init__(**kwargs)
    self.in_channels = init_ch
    self.out_channels = init_ch
    self.num_blocks = num_blocks
    self.init_ch = init_ch
    self.c1 =ConvBNRelu(init_ch)
    self.blocks = tf.keras.models.Sequential()
    for block_id in range(num_blocks):
      for layer_id in range(2):
        if layer_id == 0 :
          block = InceptionBlk(self.out_channels,strides=2)
        else:
          block = InceptionBlk(self.out_channels,strides=1)
        self.blocks.add(block)
      self.out_channels *=2
    self.p1 = tf.keras.layers.GlobalAveragePooling2D()
    self.f1 = tf.keras.layers.Dense(num_classes,activation='softmax')

  def call(self,x):
    x = self.c1(x)
    x = self.blocks(x)
    x = self.p1(x)
    y = self.f1(x)
    return  y





model = Inception10(num_blocks=2 , num_classes=10)
save_model_path = "./Cifar/InceptionNet/checkpoint/InceptionNet.ckpt"
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
