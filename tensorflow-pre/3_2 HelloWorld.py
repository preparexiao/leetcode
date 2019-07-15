#%%
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from tensorflow.examples.tutorials.mnist import input_data

#coding:utf-8
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

import tensorflow as tf
#----1.初始数据输入：interactivesession,x,W,b,y,y_
sess = tf.InteractiveSession()#创建一个interactivesesision
x = tf.placeholder(tf.float32, [None, 784])#palceholder即输入数据的地方
#tf.float32数据类型，[None,784]代表tensor的shape，None表示不限条数的输入；
# 784表示输入784维的向量。

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)#归一化 y=Wx+b
#tf.nn包含大量神经网络组件
y_ = tf.placeholder(tf.float32, [None, 10])

#----2.定义损失函数，此处用交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#reduce_mean是对每个batch数据结果求均值

#----3.定义优化函数，此处用交叉熵作为优化目标
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#定义一个优化器，学习率为0.5，优化目标设定为cross_entropy

#----4.1. 运行初始化
tf.global_variables_initializer().run()
#全局参数初始化器，并执行run()

#----4.2.开始训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #从训练集中抽取100调样本构成一个mini-batch
    train_step.run({x: batch_xs, y_: batch_ys})
    #调用train_step对样本进行训练

#----5.对模型的准确率进行验证
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#argmax求各个预测的数字中概率组大的哪一个，tf.argmax(y_, 1)赵样本的真是数字类别
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#tf.cast将correct_prediction输出的bool值转换位float32，再求平均
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
#测试数据的特征和lable输入accuracy.
