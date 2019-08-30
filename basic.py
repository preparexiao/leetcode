import tensorflow as tf
import numpy as np
b=tf.Variable(tf.zeros([100]))#生成100维的向量，初始化为0
W=tf.Variable(tf.random_uniform([784,100],-1,1))#生成784x100的随机矩阵W
x=tf.placeholder(name="x")
relu=tf.nn.relu(tf.matmul(W,x)+b)#Relu(Wx+b)
C=[...]
s=tf.Session()
for step in range(0,10):
    print("test")
    # input=np.matrix([[1,2,3],[4,5,6],[5,7,8],[7,9,11],[8,10,13],10,11,12],])
    #input=...construct 100-D input array...#为输入创建一个100维的向量
    # result=s.run(C,feed_dict={x:input})#获取Cost,供给输入x
    # print(step,result)
