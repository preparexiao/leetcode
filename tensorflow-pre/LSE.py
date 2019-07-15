import numpy as np
SIZE_COL = 4
SIZE_ROW = 1
def function(x:float)->float:
    return 2*x-1

def generate():#产生y=2x-1的随机输入，输出
    np.random.seed(1)
    x_input = np.random.uniform(0, 10, size=[SIZE_ROW, SIZE_COL])
    # print(x_input)
    y_output = np.random.uniform(0, 1, size=[SIZE_ROW, SIZE_COL])
    # print(x_input)
    for i in range(SIZE_ROW):
        for j in range(SIZE_COL):
            y_output[i][j] = y_output[i][j] + function(x_input[i][j])
            # print(y_output[i][j])
    return x_input,y_output

def add_bias(x_input,row,col):
    add_row=np.ones([row,col])
    x_input=np.row_stack((x_input,add_row))
    return x_input

def train(x_input,y_output):
    w = np.zeros([2, 1])  # y=w1x+w2
    x_bias_input = add_bias(x_input, SIZE_ROW, SIZE_COL)

    result = np.dot(x_bias_input, x_bias_input.T)
    reverse_x = np.linalg.inv(result)
    result2 = np.dot(reverse_x, x_bias_input)
    result3 = np.dot(result2, y_output.T)
    return result3

x_input,y_output=generate()
w=train(x_input,y_output)
print("函数关系为：y=",w[0][0],"x+",w[1][0])
