import numpy as np
# import tensorflow as tf
"""
a = tf.constant([1,2,3,1,1])
b = tf.constant([0,1,3,4,5])
# tf.where()函數 條件判斷 如果條件爲真 返回a 否則返回b
c = tf.where(tf.greater(a,b),a,b)
print(c)

# np.random.RandomState.rand() 返回一个[0:1)之间的随机数
rdm = np.random.RandomState(seed=1)
a = rdm.rand() # 返回一个标量
b = rdm.rand(2,3) #返回对应维度的随机数
print("a: ",a)
print("b: ",b)

# np.vstack np.hstack 数组叠加
a = [1,2,3]
b = [4,5,6]
a = np.array(a)
b = np.array(b)
c = np.vstack((a,b))
c = np.vstack((a,b,c))
d = np.hstack((a,b))
print(c)
print(d)
"""


# np.mgrid[起始值:结束值:步长,起始值:结束值:步长,……]
# x.ravel() 将x变为一维数组，把x拉直
# np.c_[数组1,数组2,……] 使返回的间隔数值点配对
# 三者共用形成网格点

# x = np.mgrid[1:3:1]
# y = np.mgrid[2:4:0.5]
x,y = np.mgrid[1:3:0.1,1:4:0.5]
print("x:",x)
print("y:",y)
grid =np.c_[x.ravel(),y.ravel()]
print("grid:\n",grid)

x,y ,z = np.mgrid[0:10:1,0:10:2,0:8:3]
print("x:",x)
print("y:",y)
print("z:",z)
grid =np.c_[x.ravel(),y.ravel(),z.ravel()]
print(grid)

"""
神经网络的复杂度
  空间复杂度
    层数 = 隐藏层 + 输出层
    总参数 = 总w + 总b
  时间复杂度
    乘加运算的次数

动态改变学习率
  指数衰减学习率 = 初始学习率 * 学习率衰减**（当前轮数/多少轮衰减一次）
  
激活函数：
  初学者：
    首先relu
    学习率设置较小
    输入特征标准化，
      即特征以0为均值，1为标准差
    初始参数中心化，
      随机生成的参数以0为均值，sqrt（2/输入特征个数）为标准差的正态分布
"""
