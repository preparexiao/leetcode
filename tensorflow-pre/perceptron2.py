# ! /usr/bin/env python

# coding=utf-8

import numpy as np


# 感知器分类的学习

class Perceptron:
    '''
    eta:学习率
    n_iter:权重向量的训练次数
    w_:权重向量
    errors_:记录神经元判断出错的次数
    '''
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        '''
        输入训练数据X，训练神经元，X输入样本，y为样本分类
        x=[[1,2],[4,5]]
        y=[-1,1]
        '''
        # 初始化权重向量,加1是因为W0
        self.w_ = np.zeros(1 + X.shape[1])
        # print(self.w_)#w_=[0,0,0]
        self.errors_ = []
        for i in range(self.n_iter):
            errors = 0
            '''
            zip(X,y)=[[1,2,-1],[4,5,1]]
            '''
            for xi, target in zip(X, y):  # 每次迭代使用一个样本去更新W
                # 相当于update=$*(y-y'),这里使用预测的结果进行误差判断
                update = self.eta * (target - self.predict(xi))
                '''
                xi是一个向量[1,2]
                update是一个数字
                update*xi等价于
                w1'=x1*update;w2'=x2*update
                '''
                self.w_[1:] += update * xi
                self.w_[0] += update * 1
                # 打印更新的W_
                # print self.w_
                # 统计 判断的正确与否次数
                errors += int(update != 0)
                self.errors_.append(errors)
    def net_input(self, X):
        '''
        z=w0*1+w1*x1+w2x2+...+wm*xm
        其中x0=1（一般w0=0,x0=1）
        '''
        return np.dot(X, self.w_[1:]) + self.w_[0] * 1
    def predict(self, X):  # 相当于sign()函数
        '''
        y>=0--->1
        y<0---->-1
        '''
        return np.where(self.net_input(X) >= 0.0, 1, -1)