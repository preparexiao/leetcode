from numpy import exp, array, random, dot,log
import numpy as np
import time
import math
class NeuralNetwork():
    def __init__(self):
        # 设置随机数种子，使每次运行生成的随机数相同
        # 便于调试
        random.seed(1)

        # 我们对单个神经元进行建模，其中有3个输入连接和1个输出连接
        # 我们把随机的权值分配给一个3x1矩阵，值在-1到1之间，均值为0。
        self.synaptic_weights_in_hide=2 * random.random((8, 8)) - 1
        self.synaptic_weights = 2 * random.random((8, 1)) - 1
        self.example_inputs = array([[0,0,0,0,0,0, 0, 1], [0,1,0,0,0,1, 1, 1], [0,0,0,1,0,1, 0, 1], [0,0,0,0,0,0, 1, 1],
                                 [0, 1, 0, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 0, 1],[0, 0, 0, 0, 0, 1, 1, 1]])
        self.training_outputs = array([[0, 1, 1, 0, 1, 1, 1, 0]])
        self.example_weight=array([[0.125],[0.125],[0.125],[0.125],[0.125],[0.125],[0.125],[0.125],[0.125]])
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
        print("test111")
        print(training_set_outputs)
        # import pdb
        # pdb.set_trace()
        for iteration in range(number_of_training_iterations):
            # 把训练集传入神经网络.
            # print("interation",training_set_inputs)
            hide_output,output = self.think(training_set_inputs)

            # 计算损失值(期望输出与实际输出之间的差。
            error = self.example_weight*(training_set_outputs - output)
            # error = training_set_outputs - output
            # print("error",error)
            www_sum=0
            for www in range(9):
                t_www=error[www][0]
                # print("www循环", www,t_www)
                self.example_weight[www][0] =math.log((1-t_www*t_www*0.5)/(t_www*t_www*0.5),math.e)
                www_sum=math.log((1-t_www*t_www*0.5)/(t_www*t_www*0.5),math.e)+www_sum

            for www in range(9):
                self.example_weight[www][0]=math.log((1-t_www*t_www*0.5)/(t_www*t_www*0.5),math.e)/www_sum
            # error = training_set_outputs - output
            sum=np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1]])
            # print(error)
            if iteration%100==0:
                print("------iteration------",iteration)
                print("Loss",np.sum(error*error*0.5))
                print("self.example_weight",self.example_weight)
            # print("training_set_outputs",training_set_outputs)
            # print("output",output)
            #     print("error",error)
            #     print("weight_error",error*self.example_weight)
                # print("self.__sigmoide_derivative",self.__sigmoid_derivative(output))
            # print("error_len",len(error))

            # 损失值乘上sigmid曲线的梯度，结果点乘输入矩阵的转置
            # 这意味着越不可信的权重值，我们会做更多的调整
            # 如果为零的话，则误区调制
            adjustment = -dot(hide_output.T, error * self.__sigmoid_derivative(output))
            adjustment_in_hide = -dot(training_set_inputs.T,
            error * self.__sigmoid_derivative(output)*self.synaptic_weights.T*self.__sigmoid_derivative(hide_output))
            # print("training_set_inputs",training_set_inputs)
            print("error",error)
            print("output",self.__sigmoid_derivative(output))
            # print("adjustment",adjustment)
            # 调制权值

            self.synaptic_weights -= adjustment
            self.synaptic_weights_in_hide-=adjustment_in_hide
            if iteration % 100 == 0:
                self.gradient_check(training_set_inputs,training_set_outputs,adjustment,adjustment_in_hide)
            # 神经网络的“思考”过程
    def think(self, inputs):
        # 把输入数据传入神经网络
        # print("think",dot(inputs,self.synaptic_weights))
        hide_output=self.__sigmoid(dot(inputs, self.synaptic_weights_in_hide))
        output=self.__sigmoid(dot(hide_output, self.synaptic_weights))
        return hide_output,output
    def gradient_check(self,training_set_inputs,training_set_outputs,adjustment,adjustment_in_hide):
        # 隐层到输出层
        gradapprox_in_to_hide=np.zeros([8,8])
        for i in range(8):
            for j in range(8):
                before = self.synaptic_weights_in_hide[i][j]
                self.synaptic_weights_in_hide[i][j] = before + 1e-7
                hide_output1, output1 = self.think(training_set_inputs)
                #计算loss
                error = training_set_outputs - output1
                loss1 = np.sum(error*error*0.5)
                self.synaptic_weights_in_hide[i][j] = before - 1e-7
                hide_output2, output2 = self.think(training_set_inputs)
                error = training_set_outputs - output2
                loss2 = np.sum(error * error * 0.5)
                #计算梯度
                gradapprox_in_to_hide[i][j] = (loss1-loss2)/(2*1e-7)
                self.synaptic_weights_in_hide[i][j] = before
        numerator = np.linalg.norm(gradapprox_in_to_hide - adjustment_in_hide)  # Step 1'
        # print("numerator",numerator)
        # print("hide_to_output",hide_to_output)
        # print("adjustment",adjustment)
        denominator = np.linalg.norm(gradapprox_in_to_hide) + np.linalg.norm(adjustment_in_hide)  # Step 2'
        difference = numerator / denominator  # Step 3'
        print("in_to_hide_difference", difference)

        hide_to_output = np.zeros([8, 1])
        for i in range(8):
            before = self.synaptic_weights[i][0]
            self.synaptic_weights[i][0] = before + 1e-7
            hide_output1, output1 = self.think(training_set_inputs)
            # 计算loss
            error = training_set_outputs - output1
            loss1 = np.sum(error * error * 0.5)
            self.synaptic_weights[i][0] = before - 1e-7
            hide_output2, output2 = self.think(training_set_inputs)
            error = training_set_outputs - output2
            loss2 = np.sum(error * error * 0.5)
            # 计算梯度
            hide_to_output[i][0] = (loss1 - loss2) / (2 * 1e-7)
            self.synaptic_weights[i][0] = before

        numerator = np.linalg.norm(hide_to_output - adjustment)  # Step 1'
        # print("numerator",numerator)
        # print("hide_to_output",hide_to_output)
        # print("adjustment",adjustment)
        denominator = np.linalg.norm(hide_to_output) + np.linalg.norm(adjustment)  # Step 2'
        difference = numerator / denominator  # Step 3'
        print("hide_to_output_difference",difference)


if __name__ == "__main__":

    # 初始化一个单神经元的神经网络
    neural_network = NeuralNetwork()

    # 输出随机初始的参数作为参照
    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # 训练集共有四个样本，每个样本包括三个输入一个输出
    training_set_inputs = array([[0,0,0,0,0,0, 0, 1], [0,1,0,0,0,1, 1, 1], [0,0,0,1,0,1, 0, 1], [0,0,0,0,0,0, 1, 1],
                                 [0, 0, 1, 0, 1, 1, 1, 1],[0, 1, 0, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 0, 1],[0, 0, 0, 0, 0, 1, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0,1,1,1,0,0]])
    # 用训练集对神经网络进行训练
    # 迭代10000次，每次迭代对权重进行微调.
    # cir=0
    # pos=[]
    # while cir<10:
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    # 输出训练后的参数值，作为对照。
    # print("New synaptic weights after training: ")
    # print(neural_network.synaptic_weights)

    # 用新样本测试神经网络.
    print("Considering new situation [1, 0, 0,0,0,0,0,0] -> ?: ")
    hide_out, out = neural_network.think(array([1, 0, 0, 0, 0, 0, 0, 0]))
    # print(hide_out)
    print(neural_network.example_weight)
    print(out)
    # pos.append(out[0])
    # print(neural_network.synaptic_weights)
    # print(neural_network.synaptic_weights_in_hide)
    # cir = cir + 1
        # example = array(
        #     [[0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 1, 1]])
    # print("pos",pos)