import numpy as np
import torch

x = torch.arange(40, dtype=torch.float32).reshape(5, 8)
print(x)
a = x.unsqueeze(1)
print(a)
b = x.unsqueeze(0)
print(b)
c = x.unsqueeze(2)
print(c)