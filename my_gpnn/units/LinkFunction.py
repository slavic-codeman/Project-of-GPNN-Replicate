"""
Created on Oct 03, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch
import torch.nn
import ConvLSTM


class LinkFunction(torch.nn.Module):
    def __init__(self, link_def, args, device):
        super(LinkFunction, self).__init__()
        self.link_def = link_def
        self.args = args
        self.device = device
        self.learn_modules = torch.nn.ModuleList()
        self.lstm = 'lstm' in link_def
        self.init_graph_conv()

    def get_definition(self):
        return self.link_def

    def get_args(self):
        return self.args

    def forward(self, edge_features):
        last_layer_output = self.ConvLSTM(edge_features) if self.lstm else edge_features

        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        return last_layer_output[:, 0, :, :]

    def init_graph_conv(self):
        input_size = self.args['edge_feature_size']
        hidden_size = self.args['link_hidden_size']
        hidden_layers = self.args['link_hidden_layers']

        if self.lstm:
            self.ConvLSTM = ConvLSTM.ConvLSTM(input_size, hidden_size, hidden_layers)
            self.learn_modules.append(torch.nn.Conv2d(hidden_size, 1, 1))
            self.learn_modules.append(torch.nn.Sigmoid())

        else:
            if self.args.get('link_relu', False):
                self.learn_modules.append(torch.nn.ReLU())
                self.learn_modules.append(torch.nn.Dropout())
            for _ in range(hidden_layers-1):
                self.learn_modules.append(torch.nn.Conv2d(input_size, hidden_size, 1))
                self.learn_modules.append(torch.nn.ReLU())
                # self.learn_modules.append(torch.nn.Dropout())
                # self.learn_modules.append(torch.nn.BatchNorm2d(hidden_size))
                input_size = hidden_size

            self.learn_modules.append(torch.nn.Conv2d(input_size, 1, 1))
            # self.learn_modules.append(torch.nn.Sigmoid())


def main():
    pass


if __name__ == '__main__':
    main()
