#mlp
import tensorflow as tf
import numpy as np

w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))#random_normal正态分布中取值
# shape: 输出张量的形状，必选
# mean: 正态分布的均值，默认为0
# stddev: 正态分布的标准差，默认为1.0
# dtype: 输出的类型，默认为tf.float32
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
# name: 操作的名称s
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# x = tf.constant([[0.7, 0.9]])
x = tf.placeholder(tf.float32, shape=(3, 2), name="input")
# print(x)
# print(w1)
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
raw_data = [1,2,3,4,5,6,7,8,9,10,11,12,13,
            14,15,16,17,18,19,20, 21, 22,
            23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
batch_size = 3
seq_length = 2
def get_batch(raw_data, batch_size, seq_length):
    data = np.array(raw_data)
    # print("data",data)
    data_length = data.shape[0]
    iterations = (data_length - 1) // (batch_size * seq_length)
    round_data_len = iterations * batch_size * seq_length
    # print(data[:round_data_len])
    xdata = data[:round_data_len].reshape(batch_size, iterations*seq_length)
    print(xdata)
    # ydata = data[1:round_data_len+1].reshape(batch_size, iterations*seq_length)
    print("data_length",data_length,"round_data_len",round_data_len,"iterations:",iterations)
    # print("enter")
    for i in range(iterations):
        print("-------Inside-------")
        x = xdata[:, i*seq_length:(i+1)*seq_length]
        # y = ydata[:, i*seq_length:(i+1)*seq_length]
        # print(x,y)
        yield x

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
# sess.run(w1.initializer)
# print(sess.run(y,feed_dict={x: [[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))
g=get_batch(raw_data,batch_size,seq_length)
for i in range(4):
    batch_xs=next(g)
    print(sess.run(y,feed_dict={x: batch_xs}))
sess.close()


# def get_batch(example_inputs, example_outputs, batch_size, seq_length):
#     # data = np.array(example_inputs)
#     # example_outputs=
#     # print("data",data)
#     data_length = example_inputs.shape[0]
#     iterations = (data_length - 1) // (batch_size * seq_length)
#     round_data_len = iterations * batch_size * seq_length
#     # print(data[:round_data_len])
#     xdata = example_inputs[:round_data_len].reshape(batch_size, iterations * seq_length)
#     print(xdata)
#     ydata = example_outputs[1:round_data_len + 1].reshape(batch_size, iterations * seq_length)
#     print("data_length", data_length, "round_data_len", round_data_len, "iterations:", iterations)
#     # print("enter")
#     for i in range(iterations):
#         print("-------Inside-------")
#         x = xdata[:, i * seq_length:(i + 1) * seq_length]
#         y = ydata[:, i * seq_length:(i + 1) * seq_length]
#         print(x, y)
#         yield x, y
#
#
# def example_main(self):
#     print("example main")