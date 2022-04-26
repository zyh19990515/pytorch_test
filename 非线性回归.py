import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.optim import SGD

class unLinearNet(nn.Module):
    def __init__(self, input_feature, num_hidden, outputs):
        super(unLinearNet, self).__init__()
        self.hidden = nn.Linear(input_feature, num_hidden)
        self.out = nn.Linear(num_hidden, outputs)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

def train(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        output = model(inputs)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch:", epoch)
        print("output:", output)
        print("loss:", loss.item())

    return model, loss, output



if __name__ == '__main__':
    x = torch.unsqueeze(torch.linspace(-3, 3, 10000), dim=1)
    y = x.pow(4)+1*torch.rand(x.size())
    plt.scatter(x.numpy(), y.numpy(), s=0.01)
    plt.show()
    net = unLinearNet(input_feature=1, num_hidden=20, outputs=1)
    inputs = x
    targets = y
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    net, loss ,out = train(net, criterion, optimizer, 10000)
    print('final loss:', loss.item())
    plt.scatter(x.numpy(), out.detach().numpy(), s=0.01)
    plt.show()