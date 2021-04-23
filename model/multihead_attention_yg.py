import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention_YG(nn.Module):

    def __init__(self,
                 num_dimensions, #400
                 num_heads, #4
                 n_class,
                 bias=True,
                 activation=F.relu):

        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.

        Schematic:
            1\ Linear layer and split to multi heads.
            2\ Scaled dot-product attention.
            3\ Concatenate the heads.
            4\ Final linear layer.
        """
        super(MultiHeadAttention_YG, self).__init__()
        # TODO why need this? can concate z and x weight matrix
        if num_dimensions % num_heads != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(num_dimensions, num_heads))
        self.num_dimensions = num_dimensions
        self.num_heads = num_heads
        self.depth = self.num_dimensions // self.num_heads
        self.activation = activation # TODO check where the activation is used
        self.bias = bias # TODO check where the bias is used

        self.wq = nn.Linear(n_class, num_dimensions, bias)
        # k: [batch_size, 64, 1024]
        self.wk = nn.Linear(1024, num_dimensions, bias)
        self.wv = nn.Linear(1024, num_dimensions, bias)
        self.linear_o = nn.Linear(num_dimensions, num_dimensions, bias)

        # self.register_buffer()

    def forward(self, q, k, v, mask=None):

        """
        Call function of MultiHeadAttention.
        :param q: the query. shape = (batch_size, seq_len_q, None)
        :param k: the key. shape = (batch_size, seq_len_k, None)
        :param v: the value. shape = (batch_size, seq_len_v, None)
        :param mask: Padding_mask.shape = (batch_size, 1, 1, seq_len)/Lookahead_mask.shape = (seq_len, seq_len)
        :return: outputs and attention weights.
        """

        # 1\ Linear layer and split to multi heads.
        batch_size = k.size()[0]
        q = self.wq(q)  # (batch_size, seq_len_q, num_dimensions) QUESTION: is this second stage query?
        k = self.wk(k)  # (batch_size, seq_len_k, num_dimensions)
        v = self.wv(v)  # (batch_size, seq_len_v, num_dimensions)
        q = self.split_heads(q, batch_size, self.num_heads, self.depth)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size, self.num_heads, self.depth)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size, self.num_heads, self.depth)  # (batch_size, num_heads, seq_len_v, depth)

        # 2\ Scaled dot-product attention.
        # attention_outputs.shape = (batch_size, num_heads, seq_len_q, depth_v)
        # attention_weights.shape = (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_outputs, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # 3\ Concatenate the heads. # TODO: how's the head useful?
        # temp.shape = (batch_size, seq_len_q, num_heads, depth)
        # concat_attention.shape = (batch_size, seq_len_q, num_dimensions)
        temp = torch.transpose(attention_outputs, 1, 2) # (batch_size, seq_len_q, num_heads, depth_v)
        concat_attention = torch.reshape(temp, (batch_size, temp.shape[1], self.num_dimensions))

        # 4\ Final linear layer.
        # output.shape = (batch_size, seq_len_q, num_dimensions)
        outputs = self.linear_o(concat_attention)

        return outputs, attention_weights


    @staticmethod
    def split_heads(x, batch_size, num_heads, depth):
        """
        Split the last dimension into (num_heads, depth).
        Then Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        :param x: shape = (batch_size, seq_len, num_dimensions)
        :param num_heads: batch size
        :param depth: depth
        :return: shape = (batch_size, num_heads, seq_len, depth)
        """
        # print(batch_size, x.shape[1], num_heads, depth)
        # print(x.shape)
        temp = torch.reshape(x, (batch_size, x.shape[1], num_heads, depth)) # batch_size, seq_len, num_head, depth
        temp = torch.transpose(temp, 1,2) # batch_size, num_head, seq_len, depth

        return temp

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        """
        Calculate the attention weights.
        Schematic:
            1\ Calculate the matmul_qk.
            2\ Scale matmul_qk.
            3\ Add the mask to the scaled tensor.
            4\ Softmax and Weighted Summation.
        Note:
            1\ q, k, v must have matching leading dimensions.
            2\ q, k must have matching last dimensions. (depth_q = depth_v)
            3\ k, v must have matching penultimate dimensions. (seq_len_k = seq_len_v)
            4\ The mask has different shapes depending on its type (padding or look ahead),
                but it must be broadcastable for addition.
        :param q: query, shape = (batch_size, num_heads, seq_len_q, depth_q)
        :param k: key, shape = (batch_size, num_heads, seq_len_k, depth_k)
        :param v: value, shape = (batch_size, num_heads, seq_len_v, depth_v)
        :param mask: Float tensor with shape broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k).
        :return: output, attention_weights
        """

        # 1\ Calculate the matmul_qk.
        k = torch.transpose(k, 2, 3) # (batch_size, num_heads, depth_k, seq_len_k)
        matmul_qk = torch.matmul(q, k)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # 2\ Scale matmul_qk.
        dk = float(q.size()[-1])
        d = math.sqrt(dk)
        scaled_attention_logits = matmul_qk / d # alpha1,i in the paper

        # 3\ Add the mask to the scaled tensor.
        if mask is not None: # TODO check what is this?
            scaled_attention_logits += (mask * -1e9)

        # 4\ Softmax and Weighted Summation.
        # attention_weights.shape = (batch_size, num_heads, seq_len_q, seq_len_k)
        # attetion_outputs.shape = (batch_size, num_heads, seq_len_q, depth_v)
        m = nn.Softmax(dim=3)
        attention_weights = m(scaled_attention_logits) # a_hat1,i in the paper, is this the second stage query?
        attention_outputs = torch.matmul(attention_weights, v) # b1 in the paper
        return attention_outputs, attention_weights