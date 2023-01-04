import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import time

class GRU(nn.Module):
    def __init__(self, INPUT_SIZE, hidden_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=hidden_size,
            batch_first=True)
        self.InitHidden()
        self.hidden_cell = self.InitHidden()
        self.out = nn.Linear(self.gru.hidden_size * 3, 2)

    def forward(self, x, hidden_cell):
        r_out, self.hidden_cell = self.gru(x, hidden_cell)
        r_out = r_out.reshape(1, 9)
        # outputs = self.out(r_out[0, :]).unsqueeze(0)
        outputs = self.out(r_out).unsqueeze(0)

        return outputs, self.hidden_cell

    def InitHidden(self):
        # h_state = torch.rand(1, 1, self.gru.hidden_size)
        h_state = torch.rand(1, self.gru.hidden_size)

        return h_state

if __name__ == '__main__':
    steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)
    input_x = np.sin(steps)

    target_y = np.cos(steps)
    # target_y = np.cos(input_x)
    # plt.plot(steps, input_x, 'b-', label='input:sin')
    plt.plot(steps, input_x, 'b-', label='input:sin')
    plt.plot(steps, target_y, 'r-', label='target:cos')
    plt.legend(loc='best')
    plt.show()
    gru = GRU(INPUT_SIZE=1)
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    hidden_cell = gru.InitHidden()
    print(hidden_cell)

    plt.figure(1, figsize=(12, 5))
    plt.ion()

    for step in range(500):
        print("this is " + str(step) + "steps")
        start, end = step * np.pi, (step + 1) * np.pi  # 一个长度为pi的区间
        steps_1 = np.linspace(start, end, 100, dtype=np.float32)  # 在(start,end)这个区间中产生100个点
        x_np = np.sin(steps_1)
        y_np = np.cos(steps_1)
        # y_np = np.cos(x_np)
        x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(-1)
        y = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(-1)
        # gru.hidden_cell = gru.InitHidden()

        prediction, hidden_cell = gru(x, hidden_cell)
        hidden_cell = hidden_cell.data
        print(hidden_cell)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        #loss.backward()
        optimizer.step()

        plt.plot(steps_1, y_np.flatten(), 'g-')
        plt.plot(steps_1, prediction.data.numpy().flatten(), 'b-')
        plt.draw();plt.pause(0.05)
    torch.save(gru, 'lstm.pkl')
    plt.ioff()
    plt.show()