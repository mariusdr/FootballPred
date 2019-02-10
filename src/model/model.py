import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class DensePredictionNet(nn.Module):
    """
    This one does the same as the LSTMPredictioNet but uses no LSTM encoder,    
    instead it just concatenates both teams player vectors into one tensor 
    and then computes a decision based on that.
    """
    def __init__(self, input_size, hidden_size=128):
        super(DensePredictionNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.prediction = nn.Sequential(
            nn.Linear(hidden_size, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        y1 = self.shared(x1)
        y2 = self.shared(x2)
        y = self.fusion( (y1 - y2) )

        return self.prediction(y)


class DensePredictionNetWithOdds(nn.Module):
    def __init__(self, player_input_size, odds_input_size, hidden_size=128):
        super(DensePredictionNetWithOdds, self).__init__()
        self.dpn = DensePredictionNet(player_input_size, hidden_size=hidden_size)

        self.fusion = nn.Sequential(
            nn.Linear(3 + odds_input_size, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2, odds):
        yd = self.dpn(x1, x2)
        odds[torch.isnan(odds)] = 0.0
        y = self.fusion(torch.cat((yd, odds), dim=1))
        return y


class MatchHistoryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MatchHistoryEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.project = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh()
        )

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, 1)

    def forward(self, seq, hidden):
        h, c = hidden

        if len(seq) == 0:
            return h[-1, :, :]

        for i in range(len(seq)):
            seq[i] = self.project(seq[i])

        inp = torch.stack(seq, dim=0)

        ys, (h, c) = self.lstm(inp, (h, c))
        return ys[-1, :, :]

    def _init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h = h.to(device=device)
        c = c.to(device=device)
        return h, c


class RNNPredictionNet(nn.Module):
    def __init__(self, history_input_size, rnn_hidden_size=64):
        super(RNNPredictionNet, self).__init__()

        self.hist_enc = MatchHistoryEncoder(history_input_size, rnn_hidden_size)

        self.prediction = nn.Sequential(
            nn.Linear(2*rnn_hidden_size, rnn_hidden_size),
            nn.ReLU(),
            nn.Linear(rnn_hidden_size, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, hist1, hist2, hidden1, hidden2):
        h1 = self.hist_enc(hist1, hidden1)
        h2 = self.hist_enc(hist2, hidden2)

        t = torch.cat((h1, h2), dim=1)
        p = self.prediction(t)
        return p


class RNNtoDensePredictionNet(nn.Module):
    def __init__(self, players_input_size, history_input_size, dn_hidden_size=128, rnn_hidden_size=32):
        super(RNNtoDensePredictionNet, self).__init__()

        self.dpn = DensePredictionNet(players_input_size, dn_hidden_size)
        self.hist_enc = MatchHistoryEncoder(history_input_size, rnn_hidden_size)

        self.fusion = nn.Sequential(
            nn.Linear(2*rnn_hidden_size + dn_hidden_size, dn_hidden_size),
            nn.ReLU(),
            nn.Linear(dn_hidden_size, dn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.prediction = nn.Sequential(
            nn.Linear(dn_hidden_size, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2, hist1, hist2, hidden1, hidden2):
        y1 = self.dpn.shared(x1)
        y2 = self.dpn.shared(x2)
        yf = self.dpn.fusion(y1-y2)
        h1 = self.hist_enc(hist1, hidden1)
        h2 = self.hist_enc(hist2, hidden2)

        t = torch.cat((h1, yf, h2), dim=1)
        y = self.fusion(t)

        p = self.prediction(y)
        return p


class RNNtoDensePredictionNetWithOdds(nn.Module):
    def __init__(self, player_input_size, history_input_size, odds_input_size, dn_hidden_size=128, rnn_hidden_size=32):
        super(RNNtoDensePredictionNetWithOdds, self).__init__()
        self.rnn = RNNtoDensePredictionNet(
            player_input_size,
            history_input_size,
            dn_hidden_size=dn_hidden_size,
            rnn_hidden_size=rnn_hidden_size)

        self.hist_enc = self.rnn.hist_enc
        self.fusion = nn.Sequential(
            nn.Linear(3 + odds_input_size, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2, hist1, hist2, hidden1, hidden2, odds):
        yd = self.rnn(x1, x2, hist1, hist2, hidden1, hidden2)
        odds[torch.isnan(odds)] = 0.0
        y = self.fusion(torch.cat((yd, odds), dim=1))
        return y


