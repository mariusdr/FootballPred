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
        self.fc1 = nn.Linear(self.input_size, 128, bias=True)
        self.fc2 = nn.Linear(128, 1024, bias=True)

        self.lstm1 = nn.LSTMCell(1024, 1024)
        self.lstm2 = nn.LSTMCell(1024, 1024)
        self.lstm3 = nn.LSTMCell(1024, 1024)

    def forward(self, player_seq):
        h1, c1 = self._init_hidden(player_seq) 
        h2, c2 = self._init_hidden(player_seq) 
        h3, c3 = self._init_hidden(player_seq) 

        for x in player_seq:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            h1, c1 = self.lstm1(x, (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
            h3, c3 = self.lstm3(h2, (h3, c3))
            
        return h3
    
    def _init_hidden(self, seq):
        x = seq[0]
        batch_size, length = x.shape 

        h = torch.zeros(batch_size, self.hidden_size)
        c = torch.zeros(batch_size, self.hidden_size)
        return h, c


class SiamesePredictionNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers):
        super(SiamesePredictionNet, self).__init__()

        self.encoder = TeamEncoder(
            input_size=input_size,
            hidden_size=1024,
            num_hidden_layers=num_hidden_layers)
        
        #self.fc1 = nn.Linear(2 * hidden_size, 3, bias=True)
        self.fc1 = nn.Linear(2048, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 128, bias=True)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, seq1, seq2):
        y1 = self.encoder(seq1)
        y2 = self.encoder(seq2)
        
        y = torch.cat((y1, y2), 1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.softmax(self.fc3(y), dim=1)
        return y



class DensePredictionNet(nn.Module):
    def __init__(self, input_size):
        super(DensePredictionNet, self).__init__()
        self.input_size = input_size
        
        # shared layers
        self.shared1 = nn.Linear(input_size, 1024)
        self.shared2 = nn.Linear(1024, 512)

        # prediction 
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x1, x2):
        x1 = F.relu(self.shared1(x1))
        x1 = F.relu(self.shared2(x1))

        x2 = F.relu(self.shared1(x2))
        x2 = F.relu(self.shared2(x2))

        y = torch.cat((x1, x2), 1)
        y = F.relu(self.fc1(y))
        y = F.softmax(self.fc2(y), dim=1)

