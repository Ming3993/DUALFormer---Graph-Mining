import torch
import torch.nn.functional as F
from torch.nn import Linear
from model.gnns import Graph_Conv
from model.sa import TransConv
from torch_geometric.nn import GCNConv, APPNP

class DUALFormer_Model(torch.nn.Module):
    def __init__(self, input_dim,
                 hidden_dim,
                 output_dim,
                 activation,
                 num_gnns,
                 num_trans,
                 num_heads,
                 dropout_trans,
                 dropout,
                 alpha,
                 use_bn,
                 lammda=0.1,
                 GraphConv='sgc'):
        super(DUALFormer_Model, self).__init__()
        self.activation = activation()
        self.num_gnns = num_gnns
        self.layers_trans = TransConv(input_dim, hidden_dim, self.activation,
                                      num_layers=num_trans, num_heads=num_heads,
                                      alpha=alpha, dropout=dropout_trans,
                                      use_bn=use_bn, use_residual=True,
                                      use_weight=True, use_act=True)

        if GraphConv == 'sgc':
            self.convs = torch.nn.ModuleList()
            for _ in range(num_gnns):
                self.convs.append(Graph_Conv())
        elif GraphConv == 'gcn':
            self.convs = torch.nn.ModuleList()
            for _ in range(num_gnns):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif GraphConv == 'appnp':
            self.convs = APPNP(num_gnns, lammda)

        self.GraphConv = GraphConv
        self.linear_project = Linear(hidden_dim, output_dim)
        self.dropout = dropout

        self.params1 = list(self.layers_trans.parameters())
        self.params2 = list(self.linear_project.parameters())

        self.traning = True
        self.reset_parameters()

    def reset_parameters(self):

        self.layers_trans.reset_parameters()
        self.linear_project.reset_parameters()

    def forward(self, x, edge_index):

        z = self.layers_trans(x)
        if self.GraphConv in ['sgc', 'gcn']:#sgc, gcn
            for i, conv in enumerate(self.convs):
                z = conv(z, edge_index)
        else:
            z = self.convs(z, edge_index) #appnp
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.linear_project(z)

        return F.log_softmax(z, dim=1)
