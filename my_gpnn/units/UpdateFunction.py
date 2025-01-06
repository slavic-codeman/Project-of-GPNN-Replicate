"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch


class UpdateFunction(torch.nn.Module):
    def __init__(self, update_def, args, device):
        super(UpdateFunction, self).__init__()
        self.update_def = update_def
        self.args = args
        self.device = device
        self.learn_modules = torch.nn.ModuleList([])
        self.init_gru()

    # Get the name of the used update function
    def get_definition(self):
        return self.u_definition

    def get_args(self):
        return self.args

    def init_gru(self):
        node_feature_size = self.args['node_feature_size']
        message_size = self.args['message_size']
        num_layers = self.args.get('update_hidden_layers', 1)
        bias = self.args.get('update_bias', False)
        dropout = self.args.get('update_dropout', False)

        self.learn_modules.append(torch.nn.GRU(message_size, node_feature_size, num_layers=num_layers, bias=bias, dropout=dropout))

    def forward(self, h_v, m_v, args=None):
        output, h = self.learn_modules[0](m_v, h_v)
        return h

def main():
    pass


if __name__ == '__main__':
    main()
