import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim

class LSTM(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=10,
            batch_first=True
        )
        self.out = nn.Linear(10, 1)

    def forward(self, x, h_state, c_state):
        r_out, (h_state, c_state) = self.lstm(x, (h_state, c_state))
        outputs = self.out(r_out[0, :]).unsqueeze(0)
        return outputs, h_state, c_state

    def InitHidden(self):
        h_state = torch.rand(1, 1, 10)
        c_state = torch.rand(1, 1, 10)
        return h_state, c_state



if __name__ == '__main__':
    steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
    input_x = np.sin(steps)
    target_y = np.cos(steps)
    plt.plot(steps, input_x, 'b-', label='input:sin')
    plt.plot(steps, target_y, 'r-', label='target:cos')
    plt.legend(loc='best')
    plt.show()
    lstm = LSTM(INPUT_SIZE=1)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.1)
    loss_func = nn.MSELoss()
    h_state, c_state = lstm.InitHidden()
    plt.figure(1, figsize=(12, 5))
    plt.ion()

    for step in range(600):
        print("this is "+str(step)+"steps")
        start, end = step*np.pi, (step+1)*np.pi #一个长度为pi的区间
        steps_1 = np.linspace(start, end, 100, dtype=np.float32)#在(start,end)这个区间中产生100个点
        x_np = np.sin(steps_1)
        y_np = np.cos(steps_1)
        x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(-1)
        y = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(-1)
        prediction, h_state, c_state = lstm(x, h_state, c_state)
        h_state = h_state.data
        c_state = c_state.data
        loss = loss_func(prediction, y)
        print(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        plt.plot(steps_1, y_np.flatten(), 'g-')
        plt.plot(steps_1, prediction.data.numpy().flatten(), 'b-')
        plt.draw(); plt.pause(0.05)
    torch.save(lstm, 'lstm.pkl')
    plt.ioff()
    plt.show()

    print(h_state)
    print(c_state)