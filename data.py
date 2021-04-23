import numpy as np
import torch
import pandas as pd

class Data:

    def __init__(self, fasta_path, index):
        """

        :param fasta_path: path to the fasta file
        :param index: index of the peak to be evaluated
        """
        self.pos_fasta_path = fasta_path
        self.index = index

    # return list of seq and reverse complement seq
    def get_seq(self):
        seq = pd.read_csv(self.pos_fasta_path, sep=">chr*",
                                header=None, engine='python').values[1::2][:, 0][self.index]

        # print(seq)
        # reverse complement
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        complement_seq = ''
        for base in seq:
            complement_seq = complement[base] + complement_seq

        # print(complement_seq)

        return seq, complement_seq

    def seq_one_hot_tensor(self, seq):
        row_index = 0
        temp = np.zeros((len(seq), 4))
        for base in seq:
            if base == 'A':
                temp[row_index, 0] = 1
            elif base == 'T':
                temp[row_index, 1] = 1
            elif base == 'G':
                temp[row_index, 2] = 1
            elif base == 'C':
                temp[row_index, 3] = 1
            row_index += 1

        seq_tensor = torch.tensor(temp).float().permute(1, 0).unsqueeze(0)

        return seq_tensor
