import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TeamEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, dropout_prob=0.5, bidirectional=False):
        """ 
        Args:
            input_size (int): length of player vectors
            hidden_size (int): size of hidden layers
            num_hidden_layers (int): number of hidden layers in the recurrent part of the network
            dropout_prob (float): dropout probability
        """
        super(TeamEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_hidden_layers
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional

        # player vector projection embedding
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)

        self.rnn1 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_prob,
            batch_first=True,
            bidirectional=self.bidirectional)


    def forward(self, inp, hidden):
        inp = F.relu(self.fc1(inp))
        inp, hidden = self.rnn1(inp, hidden)
        return inp, hidden


    def init_hidden(self, batch_size=1):
        """
        Note that a LSTM has two 'hidden' states (hidden, cell) thus we 
        give two values here.
        """
        num_directions = 2 if self.bidirectional else 1
        arg1 = num_directions * self.num_layers

        return torch.zeros(arg1, batch_size, self.hidden_size), torch.zeros(arg1, batch_size, self.hidden_size)


class SiamesePredictionNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers):
        super(SiamesePredictionNet, self).__init__()
        
        self.encoder = TeamEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers)
        self.fc1 = nn.Linear(128, 1, bias=True)


    def forward(self, inp1, inp2):
        hidden1 = self.encoder.init_hidden()
        hidden2 = self.encoder.init_hidden()

        for x1 in inp1:
            x1 = torch.unsqueeze(x1, 0)
            y1, hidden1 = self.encoder(x1, hidden1)

        for x2 in inp2:
            x2 = torch.unsqueeze(x2, 0)
            y2, hidden2 = self.encoder(x2, hidden2)


        y1 = torch.squeeze(y1, dim=0)
        y2 = torch.squeeze(y2, dim=0)

        p1 = F.relu(y1 - y2)
        p2 = F.relu(y2 - y1)
        pd = F.relu(y1 - y2) - F.relu(y2 - y1)

        p1 = self.fc1(p1)
        p2 = self.fc1(p2)
        pd = self.fc1(pd)

        outp = torch.cat((p1, p2, pd), dim=1)
        return F.softmax(outp)
