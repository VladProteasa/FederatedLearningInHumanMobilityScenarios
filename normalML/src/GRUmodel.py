import torch
import torch.nn as nn


class GRUModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_layers = hidden_layers
        self.layer_dim = layer_dim

        # RNN layers
        self.lstm = nn.GRU(
            input_dim,
            hidden_layers,
            layer_dim,
            batch_first=True,
            dropout = dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_layers, output_dim)


    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_layers).requires_grad_().cuda()


        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.lstm(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out
