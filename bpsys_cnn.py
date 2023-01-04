import copy
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from cnn_data_loader import load_data
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        # self.relu = nn.ReLU()
        self.relu = nn.Tanh()
        # self.tanh = nn.Tanh()
        # self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.Dropout = nn.Dropout(0.5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(16),
            # nn.ReLU(),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )

        # self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc1 = nn.Linear(3*3*64, 64)
        self.fc2 = nn.Linear(64, 10)
        self.Out = nn.Linear(2, 2)

        self.out = nn.Linear(10, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)



        x = x.view(x.shape[0], -1)
        #print(x.shape)

        # x = self.relu(self.fc1(x))
        # x = self.relu(self.Dropout(self.fc1(x)))
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.Dropout(self.fc2(x)))
        x = self.relu(self.fc2(x))

        # x = self.sigmoid(self.out(x))
        x = self.out(x)
        x = self.Out(x)
        # x = F.log_softmax(x, dim=-2)
        # print(x)
        return x


def get_val_loss(model, Val):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.MSELoss().to(device)
    val_loss = []
    for (data, target) in Val:
        target = torch.stack(target)
        data, target = Variable(data).to(device), Variable(target.long()).to(device)
        output = model(data)
        target = target.reshape(1, 2)
        target = target.type(torch.float)
        #print(output.data, target.data)
        loss = criterion(output, target)
        val_loss.append(loss.cpu().item())

    return np.mean(val_loss)


def train():
    Dtr, Val, Dte = load_data()
    print('train...')
    print(device)
    epoch_num = 60
    best_model = None
    min_epochs = 5
    min_val_loss = 5
    model = cnn().to(device)
    # model = cnn().to(device).eval()
    # optimizer = optim.Adam(model.parameters(), lr=0.008)
    optimizer = optim.Adamax(model.parameters(), lr=0.008)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.MSELoss().to(device)
    t_loss = []
    for epoch in tqdm(range(epoch_num), ascii=True):
        train_loss = []
        for batch_idx, (data, target) in enumerate(Dtr, 0):
            # data, target = Variable(data).to(device), Variable(target.long()).to(device)
            # print(batch_idx)
            # print(target)
            # target = torch.Tensor(target)

            # data, target = Variable(data).to(device), Variable(target).to(device)
            data = Variable(data).to(device)
            target = torch.stack(target)
            target = Variable(target).to(device)

            # target = target.view(target.shape[0], -1)
            # print(target)
            optimizer.zero_grad()
            output = model(data)
            #print(output)
            #print(output.dtype)
            target = target.reshape(1, 2)
            target = target.type(torch.float)
            # print(target.reshape(1, 2))
            print(output)
            print(target)
            # print("\n")
            loss = criterion(output, target)
            loss.backward()
            # print(loss)
            optimizer.step()
            train_loss.append(loss.cpu().item())
            # for name, param in model.named_parameters():
            #     print(name, "   ", param)
            # print("\n\n\n")
        # validation
        val_loss = get_val_loss(model, Val)
        model.train()
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        tqdm.write('Epoch {:03d} train_loss {:.5f} val_loss {:.5f}'.format(epoch, np.mean(train_loss), val_loss))
        t_loss.append(np.mean(train_loss))
    plt.figure()
    plt.plot(t_loss)
    torch.save(best_model.state_dict(), "model/cnn.pkl")


def test():
    Dtr, Val, Dte = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.load_state_dict(torch.load(".\\model\\cnn.pkl"), False)
    model.eval()
    total = 0
    current = 0
    distants_sum = 0
    for (data, target) in Dte:
        data = data.to(device)
        target = torch.stack(target)
        target = Variable(target).to(device)
        outputs = model(data)
        # print(outputs.data)
        #predicted = torch.max(outputs.data, 1)[1].data
        target = target.reshape(1, 2)
        target = target.type(torch.float)
        # print(target.data)
        # print("\n")
        out_arr = np.array(outputs.cpu().data)
        tar_arr = np.array(target.cpu().data)
        print(out_arr[0][0]*262, " ", out_arr[0][1]*181)
        print(tar_arr[0][0]*262, " ", tar_arr[0][1]*181)
        print("\n")

        distant = np.sqrt(pow((out_arr[0][0]*262-tar_arr[0][0]*262), 2)+pow((out_arr[0][1]*181-tar_arr[0][1]*181), 2))
        distants_sum += distant
        total += 1

        #current += (predicted == target).sum()

    # print('Accuracy:%d%%' % (100 * current / total))
    print("average distant error: ", distants_sum/float(total))

if __name__ == '__main__':
    train()
    # test()

