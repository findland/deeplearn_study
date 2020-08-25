import tensorflow as tf

# x1 = tf.constant([1,2,3],dtype=tf.float32)
# x2 = tf.constant(1,dtype = tf.float32)
# x3 = tf.ones([2,3])
# x4 = tf.zeros([4,1])
# x5 = tf.fill([1,5],5)
# x6 = tf.random.normal((2,2),mean = 10,stddev = 3.0)
# x7 = tf.random.truncated_normal((2,2),mean = 10,stddev = 1.0)
# x8 = tf.random.uniform((2,2),minval=0,maxval=10)
# x9 = tf.cast(x8,dtype=tf.int32)
# x9_0_mean = tf.reduce_mean(x9,axis=0)
# x9_1_mean = tf.reduce_mean(x9,axis=1)


# min_tf = tf.reduce_min(x1)
# max_tf = tf.reduce_max(x1)
# print(min_tf,max_tf)
# print(x2)
# print(x3)
# print(x4)
# print(x5)
# print("x6:\n",x6)
# print("x7:\n",x7)
# print("x8:\n",x8)
# print("x9:\n",x9)
# print("x9_0_mean\n",x9_0_mean)
# print("x9_1_mean\n",x9_1_mean)
#
"""
tf.GradientTape()
利用with结构
可以求导函数
gradient(函数,求导对象)
"""
# with tf.GradientTape() as tape:
#   w = tf.Variable(tf.constant(3,dtype=tf.float32))
#   loss = tf.pow(w,2)
# grad = tape.gradient(loss,w)
# print(grad)



"""
枚举类型函数
enumerate
类似range的用法
返回值为：索引 元素
"""
# seq = ["one" , "two" , "three"]
# for i ,ele in enumerate(seq):
#   print(i,ele)


"""
one-hot 编码
tf.one_hot
将类型转换成 one-hot编码的形式
tf.one_hot(带转换的数据,depth = 几分类)
"""
# classes = 3
# label_name = ["one","two","three"]
# l = []
# for i ,ele in enumerate(label_name):
#   l.append(i)
# print(l)
# # lables = tf.constant([1,0,2]) # 输入元素 最小为0 最大为2
# lables = tf.constant(l) # 输入元素 最小为0 最大为2
# output = tf.one_hot(lables,depth=classes)
# print(output)

# classes = 4
# lables = tf.constant([1,0,2,3])
# output = tf.one_hot(lables,depth=classes)
# print(output)

"""
将结果利用softmax方法归一
softmax(xi) = exp(xi) / sum( exp(xi) ) 
"""
# y = tf.random.uniform([3,3],minval=0,maxval=10)
# output = tf.nn.softmax(y)
# print(y)
# print(output)

"""
自减操作
assign_sub(自减内容)
"""
# # w = tf.Variable(4)
# w = tf.Variable([4,3])
# # w = tf.Variable(tf.constant([4,5]))
# w.assign_sub([1,2])
# print(w.numpy())

"""
tf.argmax(张量名,axis=操作轴)
返回对应维度的最大值的 索引号
"""
# import numpy as np
# test = np.array([[1,2,3],[2,3,4],[5,4,3],[8,7,2]])
test = tf.constant([[1,2,3],[2,3,4],[5,4,3],[8,7,2]])
print(test)
print(tf.argmax(test,axis=0))
print(tf.argmax(test,axis=1))