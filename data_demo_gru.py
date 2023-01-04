import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
from GRU import GRU
import xlrd
import cv2
if __name__ == '__main__':
    num_std = 3
    #read data
    book = xlrd.open_workbook("E:\\code\\python\\cv2practice\\position_data\\position_20221102_1.xls")
    #book = xlrd.open_workbook(".\\ptdata\\ptdata4.xls")
    sheet = book.sheets()[0]
    nrows = sheet.nrows
    # pt1_x = sheet.col_values(2)
    pt1_x = sheet.col_values(0)
    pt1_y = sheet.col_values(1)
    del pt1_x[0]
    del pt1_y[0]
    #del pt1_x[0]
    # print(pt1_x)
    # print(len(pt1_x))
    pt1_x = np.array(pt1_x, dtype=np.float32)
    pt1_y = np.array(pt1_y, dtype=np.float32)
    #pt1_x = np.concatenate((pt1_x, pt1_x))
    # pt1_x = torch.from_numpy(pt1_x).unsqueeze(0).unsqueeze(-1)
    x_mean = np.mean(pt1_x, axis=0)
    y_mean = np.mean(pt1_y, axis=0)
    #y_mean = np.mean(pt1_y, axis=0)
    x_std = np.std(pt1_x, axis=0)
    y_std = np.std(pt1_y, axis=0)
    #y_std = np.std(pt1_y, axis=0)
    norm_data_x = (pt1_x - x_mean) / x_std
    norm_data_y = (pt1_y - y_mean) / y_std
    data_set = []
    for i in range(0, len(norm_data_x)):
        data_set.append([norm_data_x[i], norm_data_y[i]])
    data_set = np.array(data_set, dtype=np.float32)
    data_set = np.concatenate((data_set, data_set), axis=0)
    # print(data_set)
    gru = GRU(INPUT_SIZE=2, hidden_size=num_std)
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.001)
    # loss_func = nn.MSELoss()
    loss_func = nn.MSELoss()
    hidden_cell = gru.InitHidden()
    # print(hidden_cell)
    #x = np.zeros((1, 5))
    #plt.figure(1, figsize=(12, 5))
    plt.ion()
    loss_data = []
    loss_plot = []
    for step in range(0, int(len(pt1_x)/num_std)):
    #for step in range(0,2000):
        print("this is " + str(step) + "steps")
        if(step==1000):
            break
        start = step * num_std
        end = (step+1) * num_std
        steps_1 = np.linspace(start, end, num_std, dtype=np.float32)


        # x = norm_data_x[start:end]
        x = data_set[start:end]
        x = torch.from_numpy(x).squeeze(1)
        # print(x.shape)
        # y = norm_data_x[start+num_std:end+num_std]
        y = data_set[start+num_std:end+num_std]
        # y = torch.from_numpy(y).unsqueeze(0).unsqueeze(-1)
        y = torch.from_numpy(y).squeeze(1)

        prediction, hidden_cell = gru(x, hidden_cell)


        hidden_cell = hidden_cell.data

        loss = loss_func(prediction, y)
        print(prediction)
        print(y)
        loss_data.append(loss.item())
        loss_data_np = np.mean(loss_data)
        loss_plot.append(loss_data_np)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # print(loss_data_np)
        optimizer.step()
        # plt.plot(steps_1, y.flatten(), 'g-')
        # plt.plot(steps_1, prediction.data.numpy().flatten(), 'b-')
        # plt.plot(loss_plot)
        # plt.scatter(steps_1, y.flatten(), s=1, c='g')
        # plt.scatter(steps_1, prediction.data.numpy().flatten(), s=1, c='b')
        #
        # plt.draw();plt.pause(0.001)
    stepp = np.linspace(0, 2000, 2000, dtype=np.float32)
    plt.figure()
    plt.plot(loss_plot, 'b')
    # plt.ioff()
    plt.savefig('./aaaaaaaa.jpg')
    plt.show()
    cv2.waitKey(0)