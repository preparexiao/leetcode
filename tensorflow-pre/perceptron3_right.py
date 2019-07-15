
import numpy as np
import matplotlib.pyplot as plt
# https://www.csdn.net/gather_23/MtTacg5sODM5MS1ibG9n.html

w = [1,1,1]
x1 =[[1,0,1],[0,1,1],[2,0,1],[2,2,1]]
x2=[[-1,-1,1],[-1,0,1],[-2,-1,1],[0,-2,1]]
flag = False
while flag != True:
    for i in range(4):
        t = 0
        for j in range(3):
            t += w[j]*x1[i][j]
        print("t1:",t)
        if(t <= 0):
           for j in range(3):
               print("w[j]",w[j])
               w[j] +=x1[i][j]
    for i in range(4):
       t = 0
       for j in range(3):
           t += w[j]*x2[i][j]
       print("t2:",t)
       if(t >= 0):
           for j in range(3):
                print("w[j]:",w[j])
                w[j] -=x2[i][j]
    flag = True
    for i in range(4):
        t1 = 0
        t2 = 0
        for j in range(3):
            t1 += w[j]*x1[i][j]
            t2 += w[j]*x2[i][j]
        if (t1 <=0 ):
            flag =False
            break
        if(t2 >=0):
            flag = False
            break
plt.figure()
for i in range(4):
    plt.scatter(x1[i][0],x1[i][1],c = 'r',marker='o')
    plt.scatter(x2[i][0],x2[i][1],c = 'g',marker='*')
plt.grid()
p1=[-2.0,2.0]
p2=[(-w[2]+2*w[0])/w[1],(-w[2]-2*w[0])/w[1]]
plt.plot(p1,p2)
plt.show()
