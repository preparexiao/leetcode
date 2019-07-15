import numpy as np

batch_size = 4
seq_length = 3
raw_data = [1,2,3,4,5,6,7,8,9,10,11,12,13,
            14,15,16,17,18,19,20, 21, 22,
            23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
def get_batch(raw_data, batch_size, seq_length):
    data = np.array(raw_data)
    # print("data",data)
    data_length = data.shape[0]
    iterations = (data_length - 1) // (batch_size * seq_length)
    round_data_len = iterations * batch_size * seq_length
    # print(data[:round_data_len])
    xdata = data[:round_data_len].reshape(batch_size, iterations*seq_length)
    print(xdata)
    ydata = data[1:round_data_len+1].reshape(batch_size, iterations*seq_length)
    print("data_length",data_length,"round_data_len",round_data_len,"iterations:",iterations)
    # print("enter")
    for i in range(iterations):
        print("-------Inside-------")
        x = xdata[:, i*seq_length:(i+1)*seq_length]
        y = ydata[:, i*seq_length:(i+1)*seq_length]
        # print(x,y)
        yield x, y
print("-------END--------")
g=get_batch(raw_data,batch_size,seq_length)
for i in range(5):
    print("--------MAIN--------")
    x,y=next(g)
    print("x",x)
    # print("y",y)