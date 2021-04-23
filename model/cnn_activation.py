import torch
from torch import nn
import torch.nn.functional as F


class CnnActivation(nn.Module):
    def __init__(self, n_class):

        super(CnnActivation, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(960, n_class)

    def forward(self, input):
        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 4, 1000]
        # Output Tensor Shape: [batch_size, 320, 993]
        x = self.Conv1(input)
        x = F.relu(x)
       # print(x.shape)

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 320, 993]
        # Output Tensor Shape: [batch_size, 320, 248]
        x = self.Maxpool(x)
        x = self.Drop1(x)
      #  print(x.shape)

        # Convolution Layer 2
        # Input Tensor Shape: [batch_size, 320, 248]
        # Output Tensor Shape: [batch_size, 480, 241]
        x = self.Conv2(x)
        x = F.relu(x)
       # print(x.shape)

        # Pooling Layer 2
        # Input Tensor Shape: [batch_size, 480, 241]
        # Output Tensor Shape: [batch_size, 480, 60]
        x = self.Maxpool(x)
        x = self.Drop1(x)
      #  print(x.shape)

        # Convolution Layer 3
        # Input Tensor Shape: [batch_size, 480, 53]
        # Output Tensor Shape: [batch_size, 960, 53]
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
       # print(x.shape)

        # Input Tensor Shape: [batch_size, 960, 53]
        # Output Tensor Shape: [batch_size, 960, 1]
        x = torch.mean(x, 2)

        #print(x.shape)

        x = self.Linear1(x)

        return x