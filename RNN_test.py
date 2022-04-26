import numpy as np
import torch
import matplotlib.pyplot as plt
from RNN import LSTM


step_test = np.linspace(0, 2*np.pi, 100, dtype=np.float32)
x = np.sin(step_test)
h_state = torch.tensor([[[ 0.3495, -0.3976,  0.3760,  0.4059, -0.3000, -0.0417,  0.4523,
           0.4213,  0.3077,  0.2339,  0.0230,  0.2873, -0.5403,  0.3409,
          -0.5259,  0.1929, -0.1601, -0.5416, -0.5180,  0.1239]]])
c_state = torch.tensor([[[ 0.6242, -0.6964,  0.7017,  0.8133, -0.6195, -0.1262,  0.7964,
           1.0260,  0.6436,  0.3777,  0.0590,  0.5212, -1.0677,  0.8184,
          -0.8391,  0.4456, -0.3221, -1.1752, -0.9891,  0.2457]]])
# h_state = torch.tensor([[[0.,  0., 0., 0.,  0.,  0.,  0.,
#           0.,  0., 0.,  0.,  0.,  0., 0.,
#            0., 0.,  0.,  0.,  0.,  0.]]])
# c_state = torch.tensor([[[0.,  0., 0., 0.,  0.,  0.,  0.,
#           0.,  0., 0.,  0.,  0.,  0., 0.,
#            0., 0.,  0.,  0.,  0.,  0.]]])

x = torch.from_numpy(x).unsqueeze(0).unsqueeze(-1)
print(x)
lstm = torch.load('D:\\code\\python\\pytorch_test\\lstm.pkl')
predict = lstm(x, h_state, c_state)
print(predict)
plt.plot(step_test, predict[0].data.numpy().flatten(), 'r-')
plt.plot(step_test[0:20], predict[1].data.numpy().flatten(), 'g-')
plt.plot(step_test, np.cos(step_test), 'b-')
plt.show()
