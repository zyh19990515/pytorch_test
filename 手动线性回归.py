import torch
import numpy as np
import matplotlib.pyplot as plt

learning_rate=0.05

#1、准备数据
x = torch.rand([500,1])
#print(x)
y_true = x*3 + 0.8

#2、通过模型计算y_predict,y=w*x+b
w = torch.rand([1,1], requires_grad=True)
b = torch.tensor(0, requires_grad=True,dtype=torch.float32)


#3、计算loss


#4、通过循环，反向传播，更新参数
for i in range(500):
    y_predict = torch.matmul(x, w) + b
    loss = (y_true - y_predict).pow(2).mean()

    if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()

    loss.backward()
    w.data = w.data - learning_rate*w.grad
    b.data = b.data - learning_rate*b.grad
    print("w,b,loss", w.item(), b.item(), loss.item())
plt.figure(figsize=(20,8))
plt.scatter(x.numpy().reshape(-1),y_true.numpy().reshape(-1))
y_predict = torch.matmul(x, w) + b
plt.plot(x.numpy().reshape(-1), y_predict.detach().numpy().reshape(-1),c='r')
plt.show()
