import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import xlrd
from GRU import GRU
import matplotlib.pyplot as plt
import xlwt

if __name__ == '__main__':
    book = xlrd.open_workbook(".\\ptdata\\ballpt_4.csv")
    predictBook = xlwt.Workbook(encoding='utf-8')
    sheet1 = predictBook.add_sheet("1")
    sheet = book.sheets()[0]
    nrows = sheet.nrows
    pt1_x = sheet.col_values(8)
    pt1_y = sheet.col_values(9)
    del pt1_x[0]
    del pt1_y[0]
    pt1_x = np.array(pt1_x, dtype=np.float32)
    pt1_x = np.concatenate((pt1_x, pt1_x))
    pt1_y = np.array(pt1_y, dtype=np.float32)
    pt1_y = np.concatenate((pt1_y, pt1_y))
    num = 2
    x_mean = np.mean(pt1_x, axis=0)
    y_mean = np.mean(pt1_y, axis=0)
    x_std = np.std(pt1_x, axis=0)
    y_std = np.std(pt1_y, axis=0)
    norm_data_x = (pt1_x - x_mean)/x_std
    norm_data_y = (pt1_y - y_mean)/y_std

    train_x = [norm_data_x[i * num:(i * num) + num] for i in range(0, int(len(norm_data_x) / num))]
    train_y = [norm_data_y[i * num:(i * num) + num] for i in range(0, int(len(norm_data_y) / num))]
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    gru = GRU(INPUT_SIZE=num)
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()
    data_loader = DataLoader(TensorDataset(train_y), batch_size=1)
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.02)
    criterion = torch.nn.MSELoss()
    loss_data = []
    hidden_cell = gru.InitHidden()
    steps = []
    lossdata = []
    cnt=0
    s = []
    outData = []
    for i, _data in enumerate(data_loader):
        s.append(cnt)
        steps = np.linspace(0, (i+1)*num, num)
        _train_x = _data[0]
        prediction, hidden_cell = gru(_train_x, hidden_cell)
        print(prediction)
        outData.append(abs(prediction.item()*x_std-x_mean))
        hidden_cell = hidden_cell.data
        optimizer.zero_grad()
        loss = criterion(prediction, _train_x)
        loss_data.append(loss.data.numpy())
        loss_np = np.mean(loss_data)
        lossdata.append(loss_np)
        loss.backward(retain_graph=True)
        optimizer.step()
        print(loss_np)
        cnt += 1
        #plt.plot(steps, _train_x.data.numpy().flatten(), 'g-')
        #plt.plot(steps, prediction.data.numpy().flatten(), 'b-')

        #plt.scatter(steps_1, y.flatten(), s=1, c='g')
        # plt.scatter(steps_1, prediction.data.numpy().flatten(), s=1, c='b')
    plt.figure()
    print(s[-1], lossdata[-1])

    plt.ylabel("test loss")
    plt.plot(s, lossdata, 'g-')
    plt.annotate(str(lossdata[-1]), xy=(s[-1], lossdata[-1]), xytext=(2600, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.0001))
    #plt.draw();plt.pause(0.05)

    #plt.ioff()
    plt.show()
    for i in range(len(outData)):
        sheet1.write(i, 0, outData[i])
    predictBook.save('y_data.csv')