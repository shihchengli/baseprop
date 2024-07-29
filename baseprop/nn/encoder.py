from lightning.pytorch.core.mixins import HyperparametersMixin
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv

from baseprop.nn.utils import get_activation_function


class GCN(nn.Module, HyperparametersMixin):
    """
    Graph Convolutional Network (GCN) with specified number of GCN and fully connected layers.

    Parameters
    ----------
    n_features : int
        Number of features in the input layer.
    hidden_channels : int, optional
        Number of hidden channels in the GCN and fully connected layers, by default 300.
    output_channels : int, optional
        Number of output channels in the final layer, by default 1.
    dropout : float, optional
        Dropout rate, by default 0.
    num_gcn_layers : int, optional
        Number of GCN layers, by default 2.
    num_ffn_layers : int, optional
        Number of fully connected layers, by default 3.
    """

    def __init__(
        self,
        n_features=-1,
        hidden_channels=300,
        dropout=0,
        num_gcn_layers=2,
        batch_norm=False,
        activation="relu",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__
        self.hidden_channels = hidden_channels
        self.dropout = nn.Dropout(dropout)
        self.tau = get_activation_function(activation)
        self.num_gcn_layers = num_gcn_layers

        # Create GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            if i == 0:
                self.gcn_layers.append(GCNConv(n_features, hidden_channels, cached=False))
            else:
                self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels, cached=False))
            if batch_norm:
                self.gcn_layers.append(BatchNorm(hidden_channels))

    def forward(self, bmgs):
        x, edge_index = bmgs.V, bmgs.edge_index

        # GCN layers
        for i in range(self.num_gcn_layers):
            layer = self.gcn_layers[i]
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)

        return x
