"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch
import torch.nn
import torch.autograd


class MessageFunction(torch.nn.Module):
    def __init__(self, message_def, args, device):
        super(MessageFunction, self).__init__()
        self.message_def = message_def
        self.args = args
        self.device = device

        self.edge = 'edge' in self.message_def
        self.concat = 'concat' in self.message_def
        self.relu = 'relu' in self.message_def
        self.init_linear()

    # Get the name of the used message function
    def get_definition(self):
        return self.message_def

    # Get the message function arguments
    def get_args(self):
        return self.args

    def init_linear(self):
        edge_feature_size = self.args['edge_feature_size']
        node_feature_size = self.args['node_feature_size']
        message_size = self.args['message_size']//2 if self.concat else self.args['message_size']

        self.edge_func = torch.nn.Linear(edge_feature_size, message_size, bias=True)
        self.node_func = torch.nn.Linear(node_feature_size, message_size, bias=True)
        self.relu = torch.nn.ReLU()

    # Message from h_v to h_w through e_vw
    def forward(self, h_v, h_w, e_vw):
        message = torch.zeros(e_vw.shape[0], self.args['message_size'], e_vw.shape[2]).to(self.device)

        for i_node in range(e_vw.shape[2]):
            edge_output = self.edge_func(e_vw[:, :, i_node])
            node_output = self.node_func(h_w[:, :, i_node])
            if self.concat:
                message[:, :, i_node] = torch.cat([edge_output, node_output], 1)
            else:
                message[:, :, i_node] = edge_output
                if not self.edge:
                    message[:, :, i_node] +=  node_output
            if self.relu:
                message[:, :, i_node] = self.relu(message[:, :, i_node])
        return message


def main():
    pass


if __name__ == '__main__':
    main()
