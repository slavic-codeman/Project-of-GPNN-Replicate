"""
Created on Oct 03, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch
import torch.nn as nn

# TODO : different
from .ConvLSTM import ConvLSTM


class LinkFunction(nn.Module):
    def __init__(self, link_def, args, device=None):
        super().__init__()
        self.l_definition = ''
        self.l_function = None
        self.learn_args = nn.ParameterList([])
        self.learn_modules = nn.ModuleList([])
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__set_link(link_def, args)

    def forward(self, edge_features):
        edge_features = edge_features.to(self.device)
        return self.l_function(edge_features)

    def __set_link(self, link_def, args):
        self.l_definition = link_def.lower()
        self.args = args

        self.l_function = {
            'graphconv': self.l_graph_conv,
            'graphconvlstm': self.l_graph_conv_lstm,
        }.get(self.l_definition)

        if self.l_function is None:
            raise ValueError(f"Incorrect definition for link function: {link_def}")

        init_parameters = {
            'graphconv': self.init_graph_conv,
            'graphconvlstm': self.init_graph_conv_lstm,
        }.get(self.l_definition, lambda: None)

        init_parameters()

    def get_definition(self):
        return self.l_definition

    def get_args(self):
        return self.args

    # Definition of linking functions
    # GraphConv
    def l_graph_conv(self, edge_features):
        last_layer_output = edge_features
        for layer in self.learn_modules:
            
            last_layer_output = layer(last_layer_output)
        return last_layer_output[:, 0, :, :]

    def init_graph_conv(self):
        input_size = self.args['edge_feature_size']
        hidden_size = self.args['link_hidden_size']

        if self.args.get('link_relu', False):
            self.learn_modules.append(nn.ReLU().to(self.device))
            self.learn_modules.append(nn.Dropout().to(self.device))
        for _ in range(self.args['link_hidden_layers'] - 1):
            self.learn_modules.append(nn.Conv2d(input_size, hidden_size, 1).to(self.device))
            self.learn_modules.append(nn.ReLU().to(self.device))
            input_size = hidden_size

        self.learn_modules.append(nn.Conv2d(input_size, 1, 1).to(self.device))

    # GraphConvLSTM
    def l_graph_conv_lstm(self, edge_features):
        last_layer_output = self.ConvLSTM(edge_features)

        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        return last_layer_output[:, 0, :, :]

    def init_graph_conv_lstm(self):
        input_size = self.args['edge_feature_size']
        hidden_size = self.args['link_hidden_size']
        hidden_layers = self.args['link_hidden_layers']

        self.ConvLSTM = ConvLSTM(input_size, hidden_size, hidden_layers).to(self.device)
        self.learn_modules.append(nn.Conv2d(hidden_size, 1, 1).to(self.device))
        self.learn_modules.append(nn.Sigmoid().to(self.device))


def main():
    """
    Unit test for LinkFunction on GPU
    """
    print("Testing LinkFunction on GPU...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GraphConv Test
    args_graphconv = {
        'edge_feature_size': 3,
        'link_hidden_size': 5,
        'link_hidden_layers': 2,
        'link_relu': True,
    }
    edge_features = torch.rand(2, 3, 8, 8).to(device)  # Batch size: 2, Features: 3, Spatial size: 8x8

    graph_conv = LinkFunction('graphconv', args_graphconv, device=device)
    output = graph_conv(edge_features)
    assert output.shape == (2, 8, 8)
    print("GraphConv test passed!")

    # GraphConvLSTM Test
    args_graphconvlstm = {
        'edge_feature_size': 3,
        'link_hidden_size': 5,
        'link_hidden_layers': 2,
    }
    for t in range(2):
        edge_features = torch.rand(2, 3, 8, 8).to(device)  # Batch size: 2, Features: 3, Spatial size: 8x8

        graph_convlstm = LinkFunction('graphconvlstm', args_graphconvlstm, device=device)
        output = graph_convlstm(edge_features)
        a=torch.mean(output)
        a.backward()
    assert output.shape == (2, 8, 8)
    print("GraphConvLSTM test passed!")


if __name__ == '__main__':
    main()
