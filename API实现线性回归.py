import torch
import torch.nn as nn
from torch.optim import SGD
#1、准备数据
x = torch.rand([500, 1])
y_ture = 6*x+6
#2、定义模型
class Mylinear(nn.Module):
    def __init__(self):
        super(Mylinear, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

#3、实例化模型，优化器类实例化，loss实例化
my_linear = Mylinear()
optimizer = SGD(my_linear.parameters(), 0.001)
loss_fn = nn.MSELoss()

#4、循环，进行梯度下降，参数更新
for i in range(50000):
    y_predict = my_linear(x)
    loss = loss_fn(y_predict, y_ture)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    params = list(my_linear.parameters())

    print(loss.item(), params[0].item(), params[1].item())
