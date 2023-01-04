from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import numpy as np
import os
import re
import torch
def Myloader(path):
    # img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # thresh, img_thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # img_erode = cv2.erode(img_thresh, kernel=kernal, iterations=5)
    # img_dilate = cv2.dilate(img_erode, kernel=kernal, iterations=5)
    # cv2.imshow("", img_dilate)
    # cv2.waitKey(0)
    # img_pil = Image.fromarray(img_dilate)
    return Image.open(path).convert('RGB')
    # return img_pil

def initprocess(path):
    data = []
    _path = []
    cnt = 0
    for path, dirs, files in os.walk(path):
        for img_file in files:
            nums = re.findall("\d+", img_file)
            _img = path + img_file
            _label = [float(nums[0])/262, float(nums[1])/181]
            data.append([_img, _label])

    #print(data)
    return data

class MyDataset(Dataset):
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):

        img, label = self.data[item]
        #print(label)
        img = self.loader(img)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)

def load_data():
    print("data processing...")
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.0),
        transforms.RandomVerticalFlip(p=0.0),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.8, 0.8, 0.8), std=(0.5, 0.5, 0.5),) # normalization
        # transforms.Normalize(mean=0.5, std=0.5)
    ])
    path1 = "E:\\code\\python\\cv2practice\\picture\\pic\\20221119_1\\"
    data1 = initprocess(path1)
    path2 = "E:\\code\\python\\cv2practice\\picture\\pic\\20221119_2\\"
    data2 = initprocess(path2)
    path3 = "E:\\code\\python\\cv2practice\\picture\\pic\\20221119_3\\"
    data3 = initprocess(path3)
    path4 = "E:\\code\\python\\cv2practice\\picture\\pic\\20221119_4\\"
    data4 = initprocess(path4)
    path5 = "E:\\code\\python\\cv2practice\\picture\\pic\\20221119_5\\"
    data5 = initprocess(path5)
    path6 = "E:\\code\\python\\cv2practice\\picture\\pic\\20221119_6\\"
    data6 = initprocess(path6)
    '''
    path7 = "E:\\code\\python\\cv2practice\\picture\\pic\\20221119_7\\"
    data7 = initprocess(path7)
    path8 = "E:\\code\\python\\cv2practice\\picture\\pic\\20221119_7\\"
    data8 = initprocess(path8)
    path9 = "E:\\code\\python\\cv2practice\\picture\\pic\\20221119_7\\"
    data9 = initprocess(path9)
    path10 = "E:\\code\\python\\cv2practice\\picture\\pic\\20221119_7\\"
    data10 = initprocess(path10)
    '''

    data = data1 + data2 + data3 + data4 + data5 + data6
    # path1 = "E:\\code\\python\\cv2practice\\picture\\pic\\"
    # data = initprocess(path1)
    print(len(data))
    train_data, val_data, test_data = data[:2800], data[2800:3200], data[3200:]
    train_data = MyDataset(train_data, transform=transform, loader=Myloader)
    Dtr = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=0)
    val_data = MyDataset(val_data, transform=transform, loader=Myloader)
    Val = DataLoader(dataset=val_data, batch_size=1, shuffle=True, num_workers=0)
    test_data = MyDataset(test_data, transform=transform, loader=Myloader)
    Dte = DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=0)
    return Dtr, Val, Dte

if __name__ == '__main__':
    # path1 = "E:\\code\\python\\cv2practice\\picture\\pic\\20221111_1\\"
    # data1 = initprocess(path1)
    # print(data1)
    # test1 = []
    # for i in data1:
    #     test1.append(i[1])
    # print(test1)
    # Dtr, Val, Dte = load_data()
    # for index, (data, target) in enumerate(Dtr, 0):
    #     target = torch.stack(target)
    #     print(target.shape)
    load_data()
