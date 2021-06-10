"""

Encoder-decoder architecture using RNN

"""


from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # do not need the embedding layer because we do not need to capture
        # similarity between words.
        #self.embedding = nn.Embedding(input_size, hidden_size)
        #self.gru = nn.GRU(hidden_size, hidden_size)
        # RNN of 26, 256, 1
        self.rnn = nn.RNN(input_size, hidden_size)


    def forward(self, input, hidden):
        #print("input")
        #print(input.size())
        embedded = input.view(1,1,-1)
        output = embedded

        #print("output")
        #print(output.size())
        #print("hidden")
        #print(hidden.size())
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, dropout_p):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(output_size, hidden_size)
        #self.gru = nn.GRU(hidden_size, hidden_size)
        self.rnn = nn.RNN(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = input.view(1, 1, -1)
        #output = F.relu(output)
        #print("input decoder")
        #print(input.size())
        #print("output decoder")
        #print(output.size())
        #print("hidden decoder")
        #print(hidden.size())
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size, device=device)
