import torch
import torch.nn as nn


class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        self.hidden_layers = hidden_layers
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(
            input_dim,
            hidden_layers,
            layer_dim,
            batch_first=True,
            dropout = dropout_prob
        )
        
        self.fc = nn.Linear(hidden_layers, output_dim)


    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_layers).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_layers).requires_grad_()

        h0 = h0.cuda()
        c0 = c0.cuda()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = out[:, -1, :]

        out = self.fc(out)
        return out
