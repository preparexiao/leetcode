#coding:utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import collections
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# for step in range(num_steps):
#     batch_inputs, batch_labels = generate_batch(
#         batch_size, num_skips, skip_window)
#     feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
def generate_batch(input,target):
    """
    函数用于产生待训练数据
    :param input:input是n*20维向量
    :param target:target是目标二分类
    :return:产生待训练数据batch,labels
    """
    batch = None
    labels = None
    return batch, labels
def network(batch_xs,target):
    # ----1.初始数据输入：interactivesession,x,W,b,y,y_
    sess = tf.InteractiveSession()  # 创建一个interactivesesision
    x = tf.placeholder(tf.float32, [None, 20])  # palceholder即输入数据
    # tf.float32数据类型，[None,20]代表tensor的shape

    W = tf.Variable(tf.zeros([20, 2]))
    b = tf.Variable(tf.zeros([2]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)  #softmax y=Wx+b

    y_ = tf.placeholder(tf.float32, [None, 2])

    # ----2.定义损失函数，此处用交叉熵
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    # 对每个batch数据结果求均值

    # ----3.定义优化函数，此处用交叉熵作为优化目标
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # 定义一个优化器，学习率为0.5，优化目标设定为cross_entropy

    # ----4.1. 运行初始化
    tf.global_variables_initializer().run()
    # 全局参数初始化器，并执行run()

    # ----4.2.开始训练
    batch_size=100
    num_steps=1000
    for i in range(num_steps):
        batch_xs, batch_ys = generate_batch(batch_size)
        # 从训练集中抽取100调样本构成一个mini-batch
        train_step.run({x: batch_xs, y_: batch_ys})
        # 调用train_step对样本进行训练

    # ----5.对模型的准确率进行验证
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # argmax求各个预测的数字中概率组大的哪一个，tf.argmax(y_, 1)赵样本的真是数字类别
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tf.cast将correct_prediction输出的bool值转换位float32，再求平均
    # 测试数据的特征和lable输入accuracy.
    return W,b,accuracy
