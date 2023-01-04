import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
from RNN import LSTM
import xlrd

if __name__ == '__main__':
    book = xlrd.open_workbook(".\\ptdata\\ballpt.xls")
    sheet = book.sheets()[0]
    nrows = sheet.nrows
    pt1_x = sheet.col_values(0)
    pt1_y = sheet.col_values(1)
    del pt1_x[0]
    del pt1_y[0]
    pt1_x = np.array(pt1_x, dtype=np.float32)
    pt1_x = np.concatenate((pt1_x, pt1_x))
    pt1_y = np.array(pt1_y, dtype=np.float32)
    pt1_y = np.concatenate((pt1_y, pt1_y))

    lstm = LSTM(INPUT_SIZE=1)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.1)
    loss_func = nn.MSELoss()
    h_state, c_state = lstm.InitHidden()
    plt.figure(1, figsize=(12, 5))
    plt.ion()
    for step in range(800):
        print("this is " + str(step) + "steps")
        start = step * 10
        end = (step + 1) * 10
        steps_1 = np.linspace(start, end, 10, dtype=np.float32)

        x = pt1_x[start:end]
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(-1)

        y = pt1_y[start:end]
        y = torch.from_numpy(y).unsqueeze(0).unsqueeze(-1)
        prediction, h_state, c_state = lstm(x, h_state, c_state)
        h_state = h_state.data
        c_state = c_state.data
        loss = loss_func(prediction, y)
        print(loss.data.numpy())
        #loss_data.append(loss.data.numpy())
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()

        loss.backward(retain_graph=True)

        optimizer.step()

        plt.plot(steps_1, y.flatten(), 'g-')
        plt.plot(steps_1, prediction.data.numpy().flatten(), 'b-')
        plt.draw();
        plt.pause(0.05)

    plt.ioff()
    plt.show()