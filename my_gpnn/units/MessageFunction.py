"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch
import torch.nn as nn


class MessageFunction(nn.Module):
    def __init__(self, message_def, args, device=None):
        super().__init__()
        self.m_definition = ''
        self.m_function = None
        self.args = {}
        self.learn_args = nn.ParameterList([])
        self.learn_modules = nn.ModuleList([])
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__set_message(message_def, args)

    # Message from h_v to h_w through e_vw
    def forward(self, h_v, h_w, e_vw, args=None):
        h_v, h_w, e_vw = h_v.to(self.device), h_w.to(self.device), e_vw.to(self.device)
        return self.m_function(h_v, h_w, e_vw, args)

    # Set a message function
    def __set_message(self, message_def, args):
        self.m_definition = message_def.lower()
        self.args = args

        self.m_function = {
            'linear': self.m_linear,
            'linear_edge': self.m_linear_edge,
            'linear_concat': self.m_linear_concat,
            'linear_concat_relu': self.m_linear_concat_relu,
        }.get(self.m_definition)

        if self.m_function is None:
            raise ValueError(f"Incorrect definition for message function: {message_def}")

        init_parameters = {
            'linear': self.init_linear,
            'linear_edge': self.init_linear_edge,
            'linear_concat': self.init_linear_concat,
            'linear_concat_relu': self.init_linear_concat_relu,
        }.get(self.m_definition, lambda: None)

        init_parameters()

    # Get the name of the used message function
    def get_definition(self):
        return self.m_definition

    # Get the message function arguments
    def get_args(self):
        return self.args

    # Definition of message functions
    def m_linear(self, h_v, h_w, e_vw, args):
        batch_size, _, num_nodes = e_vw.size()
        message = torch.zeros(batch_size, self.args['message_size'], num_nodes, device=self.device)

        for i_node in range(num_nodes):
            message[:, :, i_node] = self.learn_modules[0](e_vw[:, :, i_node]) + self.learn_modules[1](h_w[:, :, i_node])
        return message

    def init_linear(self):
        edge_feature_size = self.args['edge_feature_size']
        node_feature_size = self.args['node_feature_size']
        message_size = self.args['message_size']
        self.learn_modules.append(nn.Linear(edge_feature_size, message_size, bias=True).to(self.device))
        self.learn_modules.append(nn.Linear(node_feature_size, message_size, bias=True).to(self.device))

    def m_linear_edge(self, h_v, h_w, e_vw, args):
        batch_size, _, num_nodes = e_vw.size()
        message = torch.zeros(batch_size, self.args['message_size'], num_nodes, device=self.device)

        for i_node in range(num_nodes):
            message[:, :, i_node] = self.learn_modules[0](e_vw[:, :, i_node])
        return message

    def init_linear_edge(self):
        edge_feature_size = self.args['edge_feature_size']
        message_size = self.args['message_size']
        self.learn_modules.append(nn.Linear(edge_feature_size, message_size, bias=True).to(self.device))

    def m_linear_concat(self, h_v, h_w, e_vw, args):
        batch_size, _, num_nodes = e_vw.size()
        message = torch.zeros(batch_size, self.args['message_size'], num_nodes, device=self.device)

        for i_node in range(num_nodes):
            concatenated = torch.cat([self.learn_modules[0](e_vw[:, :, i_node]), self.learn_modules[1](h_w[:, :, i_node])], dim=1)
            message[:, :, i_node] = concatenated
        return message

    def init_linear_concat(self):
        edge_feature_size = self.args['edge_feature_size']
        node_feature_size = self.args['node_feature_size']
        message_size = self.args['message_size'] // 2
        self.learn_modules.append(nn.Linear(edge_feature_size, message_size, bias=True).to(self.device))
        self.learn_modules.append(nn.Linear(node_feature_size, message_size, bias=True).to(self.device))

    def m_linear_concat_relu(self, h_v, h_w, e_vw, args):
        batch_size, _, num_nodes = e_vw.size()
        message = torch.zeros(batch_size, self.args['message_size'], num_nodes, device=self.device)

        for i_node in range(num_nodes):
            concatenated = torch.cat([self.learn_modules[0](e_vw[:, :, i_node]), self.learn_modules[1](h_w[:, :, i_node])], dim=1)
            message[:, :, i_node] = concatenated
        return message

    def init_linear_concat_relu(self):
        edge_feature_size = self.args['edge_feature_size']
        node_feature_size = self.args['node_feature_size']
        message_size = self.args['message_size'] // 2
        self.learn_modules.append(nn.Linear(edge_feature_size, message_size, bias=True).to(self.device))
        self.learn_modules.append(nn.Linear(node_feature_size, message_size, bias=True).to(self.device))


def main():
    """
    Unit test for MessageFunction on GPU
    """
    print("Testing MessageFunction on GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Testing "linear" message function
    args = {
        'edge_feature_size': 3,
        'node_feature_size': 2,
        'message_size': 10,
    }
    message_func = MessageFunction('linear_concat_relu', args, device=device)

    h_v = torch.rand(2, 2, 5).to(device)  # Batch size: 2, Features: 3, Nodes: 5
    h_w = torch.rand(2, 2, 5).to(device)
    e_vw = torch.rand(2, 3, 5).to(device)  # Batch size: 2, Edge Features: 4, Nodes: 5

    output = message_func(h_v, h_w, e_vw)
    assert output.shape == (2, 10, 5)
    print("MessageFunction 'linear_concat_relu' test passed!")


if __name__ == '__main__':
    main()
