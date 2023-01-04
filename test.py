import torch
import numpy as np
import xlrd
if __name__ == '__main__':
    book = xlrd.open_workbook("E:\\code\\python\\cv2practice\\position_data\\position_20221102_1.xls")
    # book = xlrd.open_workbook(".\\ptdata\\ptdata4.xls")
    sheet = book.sheets()[0]
    nrows = sheet.nrows
    # pt1_x = sheet.col_values(2)
    pt1_x = sheet.col_values(0)
    pt1_y = sheet.col_values(1)
    del pt1_x[0]
    del pt1_y[0]
    # del pt1_x[0]
    print(pt1_x)
    print(len(pt1_x))
    pt1_x = np.array(pt1_x, dtype=np.float32)
    pt1_y = np.array(pt1_y, dtype=np.float32)
    # pt1_x = np.concatenate((pt1_x, pt1_x))
    # pt1_x = torch.from_numpy(pt1_x).unsqueeze(0).unsqueeze(-1)
    x_mean = np.mean(pt1_x, axis=0)
    y_mean = np.mean(pt1_y, axis=0)
    # y_mean = np.mean(pt1_y, axis=0)
    x_std = np.std(pt1_x, axis=0)
    y_std = np.std(pt1_y, axis=0)
    # y_std = np.std(pt1_y, axis=0)
    norm_data_x = (pt1_x - x_mean) / x_std
    norm_data_y = (pt1_y - y_mean) / y_std
    data_set = []
    for i in range(0, len(norm_data_x)):
        data_set.append([norm_data_x[i], norm_data_y[i]])
    data_set = np.array(data_set, dtype=np.float32)

    for i in range(0, 4):
        print(data_set[i])
