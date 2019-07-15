import tensorflow as tf
# 定义变量
w1 = tf.Variable(tf.random_normal([2,3],stddev=1),name="w1")
w2 = tf.Variable(tf.random_normal([3,1],stddev=1),name="w2")
biases1 = tf.Variable(tf.zeros([3]),name="b1")   # 隐藏层的偏向bias    [ 0. 0. 0.]
biases2 = tf.Variable(tf.zeros([1]),name="b1")   # 输出层的偏向bias   [0.]
x = tf.constant([[0.7,0.9]])
# 定义前向传播
a = tf.matmul(x,w1) + biases1
y = tf.matmul(a,w2) + biases2
# 调用会话函数输出
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y))