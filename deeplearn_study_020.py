import tensorflow as tf
import tensorflow.keras.layers
import os
import matplotlib.pyplot as plt
import numpy as np

inputwords = 'abcde'
w_2_id = {'a':0,'b':1,'c':2,'d':3,'e':4}
# id_2_onehot1 = {0:[1.,0.,0.,0.,0.],
#                1:[0.,1.,0.,0.,0.],
#                2:[0.,0.,1.,0.,0.],
#                3:[0.,0.,0.,1.,0.],
#                4:[0.,0.,0.,0.,1.]}
a = [0.,0.,0.,0.,0.]
id_2_onehot={}
for i in range(5):
  tmp = a.copy()
  tmp[i] = 1.
  id_2_onehot[i] = tmp
x_train = ['abcd','bcde','cdea','deab','eabc']
y_train = ['e','a','b','c','d']
res = []
for i in x_train:
  tmp = []
  for j in i:
    tmp.append(id_2_onehot[w_2_id[j]])
  res.append(tmp)
x_train =res
res = []
for i in y_train:
  res.append(w_2_id[i])
y_train=res

"""
x_train = [id_2_onehot[w_2_id['a']],
           id_2_onehot[w_2_id['b']],
           id_2_onehot[w_2_id['c']],
           id_2_onehot[w_2_id['d']],
           id_2_onehot[w_2_id['e']]]

y_train = [w_2_id['b'],w_2_id['c'],w_2_id['d'],w_2_id['e'],w_2_id['a']]
"""
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

# xtrain 的格式 [送入样本数,循环核展开步数，每个时间步输入的特征数]
x_train = np.reshape(x_train,(len(x_train),4,5))
y_train = np.array(y_train)

model = tf.keras.models.Sequential([
  tf.keras.layers.SimpleRNN(5),
  tf.keras.layers.Dense(5,activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics='sparse_categorical_accuracy')
save_model_path = "./RNN/RNN_onehot_four/checkpoint/RNN_onehot_four.ckpt"
if os.path.exists(save_model_path+'.index'):
  model.load_weights(save_model_path)
callback = tf.keras.callbacks.ModelCheckpoint(save_model_path,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              monitor='loss')
# fit 中没有测试集，不计算准确率，根据loss保存最优模型

history = model.fit(x_train,y_train,batch_size=32,epochs=100,
          callbacks=[callback])
model.summary()

# 作图
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1,2,1)
plt.plot(acc,label='Training ACC')
plt.title("Train ACC")
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label='Training loss')
plt.title('Train loss')
plt.legend()
plt.show()

# 预测
preNum = 5
print("predict 5:")
for i in range(preNum):
  # a1 = input("input test:")
  a1 = ""
  for j in range(4):
    a1+=inputwords[(i+j)%5]
  a = []
  for j in a1:
    a.append(id_2_onehot[w_2_id[j]])
  a = np.reshape(a,(1,4,5))
  res = model.predict(a)
  pred = tf.argmax(res,axis=1)
  pred = int(pred)
  tf.print(a1 + "->" + inputwords[pred])
