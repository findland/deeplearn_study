import tensorflow as tf
import os
import matplotlib.pyplot as plt

# 读取数据集
df = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test)=df.load_data()
x_train , x_test = x_train/255 , x_test/255


# ResNet的模块
class ResNetBlock(tf.keras.Model):
  def __init__(self,filters,strides=1,residual_path = False):
    super(ResNetBlock, self).__init__()
    self.filters = filters
    self.strides = strides
    self.residual_path = residual_path

    self.c1 = tf.keras.layers.Conv2D(filters,kernel_size=3,strides= strides ,padding='same',use_bias=False)
    self.b1 = tf.keras.layers.BatchNormalization()
    self.a1 = tf.keras.layers.Activation(activation='relu')

    self.c2 = tf.keras.layers.Conv2D(filters,kernel_size=3,strides=1 , padding='same',use_bias=False)
    self.b2 = tf.keras.layers.BatchNormalization()

    # 如果上述层中，出现了步长参数使维度发生变化的情况，
    # 调用下方的卷积层对原始的数据进行调整，
    # 使其维度与上层生成的维度相同从而可以进行相加的操作
    # 是否可以理解为strides参数不为1时就调用if中的卷积层
    if residual_path:
      self.down_c1 = tf.keras.layers.Conv2D(filters,kernel_size=1,strides=strides,padding='same',use_bias=False)
      self.down_b1 = tf.keras.layers.BatchNormalization()

    self.a2 = tf.keras.layers.Activation('relu')

  def call(self,inputs):
    residual = inputs
    x = self.c1(inputs)
    x = self.b1(x)
    x = self.a1(x)

    x = self.c2(x)
    y = self.b2(x)

    if self.residual_path:
      residual = self.down_c1(inputs)
      residual = self.down_b1(residual)
    out = self.a2(y+residual)
    return out

class ResNet18(tf.keras.Model):
  def __init__(self,block_list,inital_filters =64):
    super(ResNet18, self).__init__()
    self.num_block = len(block_list)
    self.out_filters = inital_filters

    self.c1 = tf.keras.layers.Conv2D(self.out_filters,kernel_size=3,padding='same',use_bias=False,
                                     strides=1,kernel_initializer='he_normal')
    self.b1 = tf.keras.layers.BatchNormalization()
    self.a1 = tf.keras.layers.Activation('relu')
    self.blocks = tf.keras.models.Sequential()

    for block_id in range(len(block_list)):
      for layer_id in range(block_list[block_id]):

        if block_id != 0 and layer_id == 0 :
          block = ResNetBlock(self.out_filters,strides=2,residual_path=True)
        else:
          block = ResNetBlock(self.out_filters,strides=1,residual_path=False)
        self.blocks.add(block)
      self.out_filters *=2
    self.p1 = tf.keras.layers.GlobalAveragePooling2D()
    self.f1 = tf.keras.layers.Dense(10)
  def call(self, inputs):
    x = self.c1(inputs)
    x = self.b1(x)
    x = self.a1(x)

    x = self.blocks(x)
    x = self.p1(x)
    y = self.f1(x)
    return y



model = ResNet18([2,2,2,2])


save_model_path = "./Cifar/ResNet/checkpoint/ResNet.ckpt"
if os.path.exists(save_model_path+'.index'):
  model.load_weights(save_model_path)
callback = tf.keras.callbacks.ModelCheckpoint(save_model_path,
                                              save_weights_only=True,
                                              save_best_only=True)

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
history = model.fit(x_train,y_train,batch_size=256,
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
