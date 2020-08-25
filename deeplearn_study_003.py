from sklearn.datasets import load_iris
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

"""
训练鸢尾花数据集
"""
# 1、 读入数据
x_data = load_iris().data
y_data = load_iris().target
# x = load_iris().data
# y_data = load_iris().target
# x_data = DataFrame(x,columns=load_iris().feature_names)
# pd.set_option("display.unicode.east_asian_width",True) # 设置显示方式
# print("x_data add index:\n",x_data)
# x_data["Type"] = y_data # 加一个分类列
# print("x_data add index:\n",x_data)

# 2、数据乱序
# 这里没有使用sklearn自带的数据集划分
# x_data 为数据 包括特征和分类
np.random.seed(116) #随机数种子
np.random.shuffle(x_data)
np.random.seed(116) #随机数种子
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 3、将数据集分成训练集和测试集
# (手动划分测试集和训练集)
# 训练集和测试集之间没有交集
  # 到最后30个为止作为训练集
x_train = x_data[:-30]
y_train = y_data[:-30]
  # 后30个数据作为测试集
x_test = x_data[-30:]
y_test = y_data[-30:]
# print(x_train)
# print(y_train,"/n---------/n",y_test)

'''
'''
# 转换x的数据类型，保证x中的数据类型一致，否则会报错
x_train = tf.cast(x_train,tf.float32)
x_test = tf.cast(x_test,tf.float32)


# 4、配成【特征，标签】 对，每次投入一小batch
# from_tensor_slices函数使输入特征值和标签值一一对应
# batch ：把数据集分批次每个批次batch组数据
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

# 5、定义神经网络中的可训练参数
# 生成神经网络：数据集共有4个特征输入 ，输出分类为3
# 生成参数维度为[4,3]
# Variable将参数标记为可训练
w1 = tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))
print(w1,b1)
lr = 0.01  # 学习率
train_loss_results = [] # 记录loss结果，作图用
test_acc = [] # 记录acc，作图用
loss_all = 0

# 6、嵌套迭代，with结构更新参数，显示当前loss
epoch = 1300
for epoch in range(epoch) : # 数据集 迭代
  for step , (x_train,y_train) in enumerate(train_db): # batch 迭代

    with tf.GradientTape() as tape:
      # 前向传播计算y
      y_pred = tf.matmul(x_train,w1)+b1
      y_pred = tf.nn.softmax(y_pred) # 归一化
      y_ = tf.one_hot(y_train,depth=3) # 分类one-hot化
      # 计算loss
      loss = tf.reduce_mean(tf.square(y_-y_pred)) # 均方差
      loss_all += loss.numpy()
    grade = tape.gradient(loss,[w1,b1])
    w1.assign_sub(lr*grade[0])
    b1.assign_sub(lr*grade[1])
  print("Epoch {},loss:{}".format(epoch,loss_all/4))
  train_loss_results.append(loss_all/4)
  loss_all = 0

  # 测试部分
  total_correct , total_number = 0,0
  for x_test,y_test in test_db:
    y = tf.matmul(x_test,w1)+b1
    y = tf.nn.softmax(y)
    pred = tf.argmax(y,axis=1) # 返回y中最大值的索引，即为分类
    # 转类型
    pred = tf.cast(pred,dtype=y_test.dtype)
    correct = tf.cast(tf.equal(pred,y_test), dtype=tf.int32)
    correct = tf.reduce_sum(correct)
    total_correct += int(correct)
    total_number += x_test.shape[0]

  acc = total_correct / total_number
  test_acc.append(acc)
  print("Test_acc",acc)
  print("-----------------------------")

# 画 loss 曲线
plt.title("Loss Function Curve") # 图片标题
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(train_loss_results,label="$Loss$")
plt.legend() #画图标
plt.show() # 画图像


# 画 loss 曲线
plt.title("ACC Curve") # 图片标题
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.plot(test_acc,label="$ACC$")
plt.legend() #画图标
plt.show() # 画图像

