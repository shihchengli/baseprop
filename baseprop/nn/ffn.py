from torch import nn

from baseprop.nn.utils import get_activation_function


class MLP(nn.Module):
    """
    Creates a neural network using nn.ModuleList to automatically adjust the number of layers.
    For each hidden layer, the number of inputs and outputs is constant.

    Parameters
    ----------
    in_dim : int
        Number of features contained in the input layer.
    out_dim : int
        Number of features input and output from each hidden layer, including the output layer.
    num_layers : int
        Number of layers in the network.
    activation : torch function, optional
        Activation function to be used during the hidden layers, by default torch.nn.ReLU()
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers,
        dropout=0.0,
        activation="relu",
        batch_norm=False,
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        activation = get_activation_function(activation)
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(activation)

            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
