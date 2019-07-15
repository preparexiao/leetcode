from numpy import exp, array, random, dot
import numpy as np

class NeuralNetwork():
    def __init__(self):
        # 设置随机数种子，使每次运行生成的随机数相同
        # 便于调试
        random.seed(1)

        # 我们对单个神经元进行建模，其中有3个输入连接和1个输出连接
        # 我们把随机的权值分配给一个3x1矩阵，值在-1到1之间，均值为0。
        self.synaptic_weights = 2 * random.random((8, 1)) - 1
        self.example_inputs = array([[0,0,0,0,0,0, 0, 1], [0,1,0,0,0,1, 1, 1], [0,0,0,1,0,1, 0, 1], [0,0,0,0,0,0, 1, 1],
                                 [0, 1, 0, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 0, 1],[0, 0, 0, 0, 0, 1, 1, 1]])
        self.training_outputs = array([[0, 1, 1, 0, 1, 1, 0, 0]])
        # print("synaptic_weights",self.synaptic_weights)

    # Sigmoid函数, 图像为S型曲线.
    # 我们把输入的加权和通过这个函数标准化在0和1之间。
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Sigmoid函数的导函数.
    # 即使Sigmoid函数的梯度
    # 它同样可以理解为当前的权重的可信度大小
    # 梯度决定了我们对调整权重的大小，并且指明了调整的方向
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # 我们通过不断的试验和试错的过程来训练神经网络
    # 每一次都对权重进行调整
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        training_set_outputs=training_set_outputs.T
        for iteration in range(number_of_training_iterations):
            print("------iteration------",iteration)
            # 把训练集传入神经网络.
            # print("interation",training_set_inputs)
            output = self.think(training_set_inputs)

            # 计算损失值(期望输出与实际输出之间的差。
            error = training_set_outputs - output
            # print("training_set_outputs",training_set_outputs)
            # print("output",output)
            # print("error",error)
            # print("error_len",len(error))

            # 损失值乘上sigmid曲线的梯度，结果点乘输入矩阵的转置
            # 这意味着越不可信的权重值，我们会做更多的调整
            # 如果为零的话，则误区调制
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            # print("training_set_inputs",training_set_inputs)
            # print("error",error)
            # print("output",self.__sigmoid_derivative(output))
            # print("adjustment",adjustment)
            # 调制权值
            self.synaptic_weights += adjustment
            # 神经网络的“思考”过程
    def think(self, inputs):
        # 把输入数据传入神经网络
        # print("think",dot(inputs,self.synaptic_weights))
        return self.__sigmoid(dot(inputs, self.synaptic_weights))



if __name__ == "__main__":

    # 初始化一个单神经元的神经网络
    neural_network = NeuralNetwork()

    # 输出随机初始的参数作为参照
    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # 训练集共有四个样本，每个样本包括三个输入一个输出
    training_set_inputs = array([[0,0,0,0,0,0, 0, 1], [0,1,0,0,0,1, 1, 1], [0,0,0,1,0,1, 0, 1], [0,0,0,0,0,0, 1, 1],
                                 [0, 1, 0, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 0, 1],[0, 0, 0, 0, 0, 1, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0,1,1,0,0]])
    # 用训练集对神经网络进行训练
    # 迭代10000次，每次迭代对权重进行微调.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    # 输出训练后的参数值，作为对照。
    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    # 用新样本测试神经网络.
    print("Considering new situation [1, 0, 0,0,0,0,0,0] -> ?: ")
    print(neural_network.think(array([1, 0, 0,0,0,0,0,0])))

    # example = array(
    #     [[0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 1, 1]])