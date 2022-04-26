import os
import torch.utils.data.dataloader
import torchvision
from torchvision.datasets import MNIST
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch
import numpy as np
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
def get_dataloader(train=True,batch_size=BATCH_SIZE):
    mnist = MNIST(root='./data', train=train,
                  transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize((0.1307,),(0.3081,))]),download=False)
    dataloader = torch.utils.data.dataloader.DataLoader(mnist,batch_size=batch_size,shuffle=True)

    return dataloader


class MymnistNet(nn.Module):
    def __init__(self):
        super(MymnistNet, self).__init__()
        self.fc1 = nn.Linear(1*28*28,28)
        self.fc2 = nn.Linear(28,10)
    def forward(self,input):
        #parameter: batch_size:1,28,28
        #1\修改形状
        x = input.view([input.size(0),1*28*28])
        #2、进行全连接操作
        x = self.fc1(x)
        #3、激活函数处理
        x = F.relu(x)
        #4、输出层
        out = self.fc2(x)
        return F.log_softmax(out, dim=-1)
model = MymnistNet()
optimizer = Adam(model.parameters(),lr=0.001)
if(os.path.exists("./model/model.pkl")):
    model.load_state_dict(torch.load("./model/model.pkl"))
    optimizer.load_state_dict(torch.load("./model/optimizer.pkl"))
def train(epoch):
    """实现训练的过程"""

    data_loader = get_dataloader()
    for idx,(input,target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(input)#调用模型，得到预测值
        loss = F.nll_loss(output,target)#得到损失，tensor
        loss.backward()#反向传播
        optimizer.step()#梯度更新
        if(idx%100==0):
            print(epoch,idx,loss.item())
        #模型保存
        if(idx%100==0):
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")

def test():
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(train=False, batch_size=TEST_BATCH_SIZE)
    for idx,(input,target) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output,target)
            loss_list.append(cur_loss)
            #计算准确率
            #output[batch_size,10] target:[batch_size]
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print("平均准确率，平均损失：",np.mean(acc_list), np.mean(loss_list))

if __name__ == '__main__':
    # for i in range(3):
    #     train(i)
    test()