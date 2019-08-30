import numpy as np
import random
SIZE_COL = 100
SIZE_ROW = 2
LearningRate=0.1
def function(x:int)->int:
    return 2*x

def generate():#产生y=2x-1的随机输入，输出
    np.random.seed(1)
    x_input = np.random.randint(-20, 20, size=[SIZE_COL,SIZE_ROW])
    # print(x_input)
    y_output = np.random.randint(-20, 20, size=[SIZE_COL, 1])
    # print(x_input)
    for i in range(SIZE_COL):
        y_output[i][0] = sign(x_input[i][1]-function((x_input[i][0])))
            # print(y_output[i][j])
    print("x_input", x_input)
    print("y_output",y_output)
    return x_input,y_output

def add_bias(x_input,row,col):
    add_row=np.ones([row,col])
    x_input=np.row_stack((x_input,add_row))
    return x_input

def sign(wx)->int:
    if wx<0:
        return -1
    else:
        return 1

def check(y_output,x_input,w)->float:
    TP=0
    print("x_input[0]",x_input[0],y_output[0])
    i=0
    while i<len(x_input):
        # print("check",np.sum(w*x_input[i]))
        if sign(np.sum(w*x_input[i]))==y_output[i][0]:
            TP=TP+1
        i=i+1
    return TP/len(x_input)

def train(x_input,y_output,iteration):
    w = np.zeros([1, 2])  # y=w1x+w2
    # import pdb
    # pdb.set_trace()
    for itrn in range(iteration):
        print("-----itrn----",itrn)
        order=range(0,len(x_input[0]))
        order=random.sample(order,len(x_input[0]))
        # print(order)
        sum_x=np.sum(x_input)
        for ex in range(len(x_input[0])):
            print("----ex-----",ex)
            rand_x_input = x_input[order[ex]]/sum_x#
            rand_y_output=y_output[order[ex]]
            print("rand_x_input1",rand_x_input)
            print("w2",w)
            loss=w[0]*rand_y_output*rand_x_input
            print("rand_x_input3",rand_x_input)
            print("rand_y_output4",rand_y_output)
            print("loss",loss)
            loss=np.sum(loss)
            if loss<0 or loss==0:
                print("success")
                print("update",LearningRate*rand_x_input*rand_y_output)
                w = w - LearningRate * rand_x_input * rand_y_output

    return w[0]


x_input,y_output=generate()
print("input",x_input,y_output)
w=train(x_input,y_output,1000)
print(check(y_output,x_input,w))
print(w)
print(y_output)
print("函数关系为：y=",w[0]/w[1],"x+")
