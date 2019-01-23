import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TeamEncoder(nn.Module):
    """
    Given a set of players (vectors of player stats) encode 
    them using an LSTM model.
    """
    def __init__(self, input_size, hidden_size):
        super(TeamEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # player vector projection embedding
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)

        self.lstm1 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        # self.dropout1 = nn.Dropout()
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        # self.dropout2 = nn.Dropout()
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)

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


class LSTMPredictionNet(nn.Module):
    """
    Given the players of two teams, encode both teams with the TeamEncoder LSTM 
    and then decide which team is more likely to win based on their encodings.
    """
    def __init__(self, input_size, hidden_size):
        super(LSTMPredictionNet, self).__init__()

        self.encoder = TeamEncoder(input_size=input_size, hidden_size=256)

        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 3)

    def forward(self, seq1, seq2):
        y1 = self.encoder(seq1)
        y2 = self.encoder(seq2)

        y = torch.cat((y1, y2), 1)
        y = F.relu(self.fc1(y))
        y = F.softmax(self.fc2(y), dim=1)
        return y



class DensePredictionNet(nn.Module):
    """
    This one does the same as the LSTMPredictioNet but uses no LSTM encoder,    
    instead it just concatenates both teams player vectors into one tensor 
    and then computes a decision based on that.
    """
    def __init__(self, input_size):
        super(DensePredictionNet, self).__init__()
        self.input_size = input_size

        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.prediction = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        x1 = self.shared(x1)
        x2 = self.shared(x2)

        y = torch.cat((x1, x2), 1)

        return self.prediction(y)


class DenseCompNet(nn.Module):
    """
    Instead of computing a probability of the game outcome (i.e. home win / away win / draw)
    this one tries to compute which team is stronger, i.e. supposed to win, without making hard 
    decisions. 

    Football has a heavy home-field bias, i.e. Bundesliga 07/08 had about 46% home wins, 
    so networks with home win / away win / draw output can just "learn" to put alot of 
    probability mass on home win and get good results w.o. really learning something ...
    """
    def __init__(self, input_size):
        super(DenseCompNet, self).__init__()
        self.input_size = input_size

        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.prediction = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 2),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        x1 = self.shared(x1)
        x2 = self.shared(x2)

        y = torch.cat((x1, x2), 1)
        return self.prediction(y)


class LSTMCompNet(nn.Module):
    """
    Same idea as above DenseCompNet but again with LSTMs instead of dense nets.
    """
    def __init__(self, input_size, hidden_size):
        super(LSTMCompNet, self).__init__()

        self.encoder = TeamEncoder(input_size=input_size, hidden_size=256)

        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 3)

    def forward(self, seq1, seq2):
        y1 = self.encoder(seq1)
        y2 = self.encoder(seq2)

        y = torch.cat((y1, y2), 1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        return y

