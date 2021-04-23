import torch
from torch import nn
from model.multihead_attention_yg import MultiHeadAttention_YG
import torch.nn.functional as F


class ChQueryDiagonal(nn.Module):

    def __init__(self, query_dim):
        super(ChQueryDiagonal, self).__init__()
        self.query_dim = query_dim
        self.diagonal = torch.eye(query_dim).unsqueeze(0).to("cuda:0")  # shape: [1, q_dim, q_dim]

    def forward(self, batch_size):
        return self.diagonal.repeat(batch_size, 1, 1)


class DeepATT(nn.Module): # no padding
    def __init__(self, q_dim):
        super(DeepATT, self).__init__()
        self.q_dim = q_dim
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=1024,
                               kernel_size=30, stride=1, padding=0)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=15, stride=15, padding=0)
        # check the LSTM
        self.BiLSTM = nn.LSTM(input_size=1024, hidden_size=512,
                              num_layers=2,
                              batch_first=True,
                              dropout=0.4,
                              bidirectional=True)

        self.category_encoding = ChQueryDiagonal(q_dim)

        self.multi_head_attention = MultiHeadAttention_YG(400, 4, self.q_dim)  # num_dimensions, num_heads

        self.Linear1 = nn.Linear(400, 100) # 400 * 100 + 100 -> 40100
        self.Linear2 = nn.Linear(100, 1) # TODO why the output feature dimention is 1 instead of 919?

        self.Drop1 = nn.Dropout(0.2)
        self.Drop2 = nn.Dropout(0.2)

    def forward(self, input):
        """
        Forward propagation of DeepAttention model.
        :param inputs: shape = (batch_size, length, c)
        :return: shape = (batch_size, q_dim)
        """
        batch_size = input.size()[0]
        #print(batch_size)

        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 4, 1000]
        # Output Tensor Shape: [batch_size, 1024, 971]
        x = self.Conv1(input)
        x = F.relu(x)
        #print(x.size()) # [batch_size, 1024, 971] correct

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 1024, 971]
        # Output Tensor Shape: [batch_size, 1024, 64]
        x = self.Maxpool1(x)
        x = self.Drop1(x) # [batch_size, 1024, 64] correct

        #print(x.size())

        x_x = torch.transpose(x, 1, 2)  # [batch_size, 64, 1024] batch_size, seq_length, feature/input_size
        #print(x_x.size()) # [batch_size, 64, 1024] correct

        # Bidirectional RNN Layer 1
        # Input Tensor Shape: [batch_size, 64, 1024] # TODO double check the dimension
        # Output Tensor Shape: [batch_size, 64, 1024]
        x, (h_n, h_c) = self.BiLSTM(x_x) # [batch_size, 64, 1024] correct

        #print(x.size())

        # Category Multi-head Attention Layer 1
        # Input Tensor Shape: v.shape = [batch_size, 64, 1024]
        #                     k.shape = [batch_size, 64, 1024]
        #                     q.shape = [batch_size, q_dim, q_dim]
        # Output Tensor Shape: temp.shape = [batch_size, q_dim, 400]
        query = self.category_encoding(batch_size)
        # query = self.query_tensor
        # print("--------------------query-------------")
        # print(query.shape)
        # print(query)
        x, _ = self.multi_head_attention(query, x, x)

        #print(query.size()) # torch.Size([batch_size, q_dim, q_dim]) correct
        #print(x.size()) # torch.Size([batch_size, q_dim, 400]) correct

        # Dropout Layer 2
        x = self.Drop2(x)

        # Category Dense Layer 1
        # Input Tensor Shape: [batch_size, q_dim, 400]
        # Output Tensor Shape: [batch_size, q_dim, 100]
        x = self.Linear1(x) # torch.Size([100, q_dim, 100]) correct
        x = F.relu(x)

        #print(x.size())

        # Category Dense Layer 2 (weight-share)
        # Input Tensor Shape: [batch_size, q_dim, 100]
        # Output Tensor Shape: [batch_size, q_dim, 1]
        output = self.Linear2(x)
        #print(output.size()) #torch.Size([100, q_dim, 1]) correct

        # Output Tensor Shape: [batch_size, q_dim]
        output = torch.reshape(output, (-1, self.category_encoding.query_dim))
        #print(output.size()) #torch.Size([100, q_dim]) correct

        return output



class DeepATT_modified(nn.Module): # no padding
    def __init__(self, q_dim):
        super(DeepATT_modified, self).__init__()
        self.q_dim  = q_dim
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=1024,
                               kernel_size=30, stride=1, padding=0)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=15, stride=15, padding=0)
        # check the LSTM
        self.BiLSTM = nn.LSTM(input_size=1024, hidden_size=512,
                              num_layers=2,
                              batch_first=True,
                              dropout=0.4,
                              bidirectional=True)

        self.category_encoding = torch.eye(self.q_dim).unsqueeze(0).to("cuda:0") # shape: [1, q_dim, q_dim]
        self.multi_head_attention = MultiHeadAttention_YG(8, 4, self.q_dim)#num_dimensions, num_heads
        #self.multi_head_attention = torch.nn.ModuleList([self.multi_head_attention]) # TODO double check: use this to convert module to cuda
        #device = torch.device('cuda:0')
        #self.multi_head_attention = self.multi_head_attention.to(device)
        self.Linear1 = nn.Linear(8, 4) # 400 * 100 + 100 -> 40100
        self.Linear2 = nn.Linear(4, 1) # TODO why the output feature dimention is 1 instead of 919?

        self.Drop1 = nn.Dropout(0.2)
        self.Drop2 = nn.Dropout(0.2)

    def forward(self, input):
        """
        Forward propagation of DeepAttention model.
        :param inputs: shape = (batch_size, length, c)
        :return: shape = (batch_size, q_dim)
        """
        batch_size = input.size()[0]
        #print(batch_size)

        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 4, 1000]
        # Output Tensor Shape: [batch_size, 1024, 971]
        x = self.Conv1(input)
        x = F.relu(x)
        #print(x.size()) # [batch_size, 1024, 971] correct

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 1024, 971]
        # Output Tensor Shape: [batch_size, 1024, 64]
        x = self.Maxpool1(x)
        x = self.Drop1(x) # [batch_size, 1024, 64] correct

        #print(x.size())

        x_x = torch.transpose(x, 1, 2)  # [batch_size, 64, 1024] batch_size, seq_length, feature/input_size
        #print(x_x.size()) # [batch_size, 64, 1024] correct

        # Bidirectional RNN Layer 1
        # Input Tensor Shape: [batch_size, 64, 1024] # TODO double check the dimension
        # Output Tensor Shape: [batch_size, 64, 1024]
        x, (h_n, h_c) = self.BiLSTM(x_x) # [batch_size, 64, 1024] correct

        #print(x.size())

        # Category Multi-head Attention Layer 1
        # Input Tensor Shape: v.shape = [batch_size, 64, 1024]
        #                     k.shape = [batch_size, 64, 1024]
        #                     q.shape = [batch_size, q_dim, q_dim]
        # Output Tensor Shape: temp.shape = [batch_size, q_dim, 400]
        query = self.category_encoding.repeat(batch_size, 1, 1)
        x, _ = self.multi_head_attention(query, x, x)

        #print(query.size()) # torch.Size([batch_size, q_dim, q_dim]) correct
        #print(x.size()) # torch.Size([batch_size, q_dim, 400]) correct

        # Dropout Layer 2
        x = self.Drop2(x)

        # Category Dense Layer 1
        # Input Tensor Shape: [batch_size, q_dim, 400]
        # Output Tensor Shape: [batch_size, q_dim, 100]
        x = self.Linear1(x) # torch.Size([100, q_dim, 100]) correct
        x = F.relu(x)

        #print(x.size())

        # Category Dense Layer 2 (weight-share)
        # Input Tensor Shape: [batch_size, q_dim, 100]
        # Output Tensor Shape: [batch_size, q_dim, 1]
        output = self.Linear2(x)
        #print(output.size()) #torch.Size([100, q_dim, 1]) correct

        # Output Tensor Shape: [batch_size, q_dim]
        output = torch.reshape(output, (-1, self.q_dim))
        #print(output.size()) #torch.Size([100, q_dim]) correct

        return output
