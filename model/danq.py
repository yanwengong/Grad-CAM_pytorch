import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class DanQ(nn.Module):
    def __init__(self, n_class):
        super(DanQ, self).__init__()
        self.n_class = n_class
        self.Conv1 = nn.Conv1d(in_channels=4,
                               out_channels=320, kernel_size=26)
        self.Maxpool = nn.MaxPool1d(kernel_size=13,
                                    stride=13)
        self.Drop1 = nn.Dropout(p=0.2)
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=320,
                              num_layers=2,
                                 batch_first=True,
                                 dropout=0.5,
                                 bidirectional=True)
        self.Linear1 = nn.Linear(75*640, 925)
        self.Linear2 = nn.Linear(925, self.n_class)

    def forward(self, input):
        x = self.Conv1(input) # torch.Size([1024, 320, 975]) batchsize, out_channels, seq_len

        x = F.relu(x) # torch.Size([1024, 320, 975])

        x = self.Maxpool(x) # torch.Size([1024, 320, 75])

        x = self.Drop1(x) # torch.Size([1024, 320, 75])

        x_x = torch.transpose(x, 1, 2) # torch.Size([1024, 75, 320]) batch_size, seq_len, input_size/num_feature

        x, (h_n,h_c) = self.BiLSTM(x_x) # torch.Size([1024, 75, 640]) batch_size, seq_len, input_size/num_feature

        #x, h_n = self.BiGRU(x_x)
        #x = x.contiguous().view(-1, 75*640) # torch.Size([1024, 48000])
        x = torch.flatten(x, 1)

        x = self.Linear1(x) #torch.Size([1024, 925]) # batchsize, out_channels

        x = F.relu(x) # torch.Size([1024, 925])

        x = self.Linear2(x) # torch.Size([1024, 8])
        #print(x.size())

        # print("one epoch done")
        #x = torch.sigmoid(x)
        return x

    def __str__(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        summary(DanQ, (4, 1000))


class Simple_DanQ(torch.nn.Module):
    def __init__(self, n_class):
        super(Simple_DanQ, self).__init__()
        self.n_class = n_class
        self.Conv = nn.Conv1d(in_channels=4,
                              out_channels=32,
                              kernel_size=26)
        self.Maxpool = nn.MaxPool1d(kernel_size=13,
                                    stride=13)
        self.Drop = nn.Dropout(0.1)
        self.BiLSTM = nn.LSTM(input_size=75, hidden_size=32,
                              num_layers=2,
                              batch_first=True,
                              dropout=0.5,
                              bidirectional=True)
        self.Linear1 = nn.Linear(32*32*2, 32)
        self.Linear2 = nn.Linear(32, self.n_class)


    def forward(self, input):
        x = self.Conv(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop(x)
        x, _ = self.BiLSTM(x)
        #print(f'output shape from LSTM layer: {x.shape}')
        x = torch.flatten(x, 1)
        #print(f'output shape from flatten layer: {x.shape}')
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        x = torch.sigmoid(x)

        return x

    def __str__(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        summary(Simple_DanQ, (4, 1000))


class Complex_DanQ(torch.nn.Module):
    def __init__(self, n_class):
        super(Complex_DanQ, self).__init__()
        self.n_class = n_class
        self.Conv1 = nn.Conv1d(in_channels=4,
                               out_channels=320, # more
                               kernel_size=30) # smaller 16, 24
        self.Conv2 = nn.Conv1d(in_channels=320,
                               out_channels=160, # smaller
                               kernel_size=12)
        self.Maxpool = nn.MaxPool1d(kernel_size=13,
                                    stride=11)
        self.Drop1 = nn.Dropout(0.1)
        self.BiLSTM = nn.LSTM(input_size=160, hidden_size=40,
                              num_layers=2,
                              batch_first=True,
                              dropout=0.4,
                              bidirectional=True)
        self.Linear1 = nn.Linear(87*40*2, 256)
        self.Linear2 = nn.Linear(256, 256)
        self.Linear3 = nn.Linear(256, self.n_class) ## TODO 0312 changed back to 1, to load the trainined model

        self.Drop2 = nn.Dropout(0.2)

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Conv2(x)
        x = F.relu(x)

        x = self.Maxpool(x)

        x = self.Drop2(x)
        x_x = torch.transpose(x, 1, 2)

        x, _ = self.BiLSTM(x_x)

        x = torch.flatten(x, 1)

        x = self.Linear1(x)

        x = F.relu(x)

        x = self.Linear2(x)
        x = F.relu(x)
        x = self.Linear3(x)
        #x = torch.sigmoid(x)

        return x

    def __str__(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        summary(Complex_DanQ, (4, 1000))



class Simple_DanQ_noLSTM(torch.nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.Conv1 = nn.Conv1d(in_channels=4,
                               out_channels=32,
                               kernel_size=30)
        self.Conv2 = nn.Conv1d(in_channels=32,
                               out_channels=16,
                               kernel_size=12)
        self.Maxpool = nn.MaxPool1d(kernel_size=13,
                                    stride=11)
        self.Drop1 = nn.Dropout(0.1)
        self.Linear1 = nn.Linear(1392, 256)
        #self.Linear2 = nn.Linear(256, 256)
        self.Linear3 = nn.Linear(256, self.n_class)


        self.Drop2 = nn.Dropout(0.2)


    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Conv2(x)
        x = F.relu(x)

        x = self.Maxpool(x)

        x = self.Drop2(x)

        x = torch.flatten(x, 1)
        #print(f'output shape from flatten layer: {x.shape}')
        x = self.Linear1(x)

        x = F.relu(x)

        #x = self.Linear2(x)
        #x = F.relu(x)
        x = self.Linear3(x)
        x = torch.sigmoid(x)
        return x

    def __str__(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        summary(Complex_DanQ, (4, 1000))

