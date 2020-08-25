import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
# 1、读取数据
iris = load_iris()
data = np.hstack((iris.data,np.array([iris.target]).T))
# print (data)
# 2、划分数据集
np.random.seed(116)
np.random.shuffle(data)
x_train = data[:-30,0:4]
y_train = data[:-30,-1]
x_test = data[-30:,0:4]
y_test = data[-30:,-1]

x_train = tf.cast(x_train,dtype=tf.float32)
y_train = tf.cast(y_train,dtype=tf.int32)
x_test = tf.cast(x_test,dtype=tf.float32)
y_test = tf.cast(y_test,dtype=tf.int32)

# 3、将数据分配成 【特征，目标】的形式
# batch(n) 是什么形式的数据
#   将一个大数据集（包含N个数据）划分成若干个小数据集
#   每个数据集中包含n个数据，小数据集中的数据不重复，
#   当最后一个数据集数据不足时则缺省
tf.random.set_seed(116)
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

# 4、建立隐藏层
w1 = tf.Variable(tf.random.truncated_normal([4,3],stddev=0.01,dtype=tf.float32))
b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.01,dtype=tf.float32))
print(w1,b1)
LR = 0.8  # 基础学习率
LR_DECAY = 0.9 # 学习率衰减率
LR_STEP = 2 # 多少轮衰减一次
train_loss_results = [] # 记录loss结果，作图用
test_acc = [] # 记录acc，作图用
loss_all = 0

# 5、训练
epoch = 200
for epoch in range(epoch):
  lr = LR * LR_DECAY **(epoch/LR_STEP)
  for step, (x_train, y_train) in enumerate(train_db):  # batch 迭代
    with tf.GradientTape() as tape:
      y_pre = tf.matmul(x_train,w1)+b1
      # print (y_pre)
      y_pre = tf.nn.softmax(y_pre) # 归一化
      # print (y_pre)
      y_train = tf.one_hot(y_train,depth=3,dtype=tf.float32) # 目标值one—hot化
      # 保持数据类型一致
      loss = tf.reduce_mean(tf.square(y_pre - y_train)) # 损失函数
      # print(loss)
      loss_all += loss.numpy()
    grade = tape.gradient(loss,[w1,b1])
    # 更新参数
    w1.assign_sub(lr*grade[0])
    b1.assign_sub(lr*grade[1])
  print("Epoch {},loss:{}".format(epoch,loss/4 ))
  train_loss_results.append(loss_all /4 )
  loss_all = 0

# 测试部分 每次迭代进行一次验证
  total_correct , total_number = 0,0
  for x_test,y_test in test_db:
    y_pre = tf.matmul(x_test, w1) + b1
    print (y_pre)
    y_pre = tf.nn.softmax(y_pre)  # 归一化
    pred = tf.argmax(y_pre, axis=1)  # 返回y中最大值的索引，即为分类
    # 转类型,为了进行比较
    pred = tf.cast(pred, dtype=y_test.dtype)

    correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
    # print(correct)
    correct = tf.reduce_sum(correct)
    total_correct += int(correct)
    total_number += x_test.shape[0]

  acc = total_correct / total_number
  test_acc.append(acc)
  print("Test_acc: ", acc)
  print("-----------------------------")

# 画 loss 曲线
plt.title("Loss Function Curve") # 图片标题
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(train_loss_results,label="$Loss$")
plt.legend() #画图标
plt.show() # 画图像


# 画 ACC 曲线
plt.title("ACC Curve") # 图片标题
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.plot(test_acc,label="$ACC$")
plt.legend() #画图标
plt.show() # 画图像
