import torch
from torch.utils.data import Dataset

data_path = r''

class MyDataset(Dataset):
    def __init__(self):
        self.lines = open(data_path).readline()

    def __getitem__(self, index):
        return self.lines[index]
        #之后进行预处理

    def __len__(self):
        return len(self.lines)