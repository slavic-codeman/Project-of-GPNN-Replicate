"""
Created on FEB 25, 2018

@author: Siyuan Qi

Description of the file.

"""

import os

import torch
import torch.nn
import torch.autograd

import units


class GPNN_VCOCO(torch.nn.Module):
    def __init__(self, model_args, device = None):
        super(GPNN_VCOCO, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_args = model_args.copy()
        if model_args['resize_feature_to_message_size']:
            # Resize large features
            self.edge_feature_resize = torch.nn.Linear(model_args['edge_feature_size'], model_args['message_size']).to(self.device)
            self.node_feature_resize = torch.nn.Linear(model_args['node_feature_size'], model_args['message_size']).to(self.device)
            torch.nn.init.xavier_normal(self.edge_feature_resize.weight)
            torch.nn.init.xavier_normal(self.node_feature_resize.weight)

            model_args['edge_feature_size'] = model_args['message_size']
            model_args['node_feature_size'] = model_args['message_size']

        self.link_fun = units.LinkFunction('GraphConv', model_args, self.device).to(self.device)
        self.sigmoid = torch.nn.Sigmoid().to(self.device)
        self.message_fun = units.MessageFunction('linear_concat_relu', model_args, self.device).to(self.device)
        self.update_fun = units.UpdateFunction('transformer' if model_args.get('transformer', False) else 'gru', model_args, self.device).to(self.device)
        self.readout_fun = units.ReadoutFunction('fc', {'readout_input_size': model_args['node_feature_size'], 'output_classes': model_args['hoi_classes']}, self.device).to(self.device)
        self.readout_fun2 = units.ReadoutFunction('fc', {'readout_input_size': model_args['node_feature_size'], 'output_classes': model_args['roles_num']}, self.device).to(self.device)

        self.propagate_layers = model_args['propagate_layers']

        self._load_link_fun(model_args)

    def forward(self, edge_features, node_features, adj_mat, node_labels, node_roles, human_nums, obj_nums, args):
        if self.model_args['resize_feature_to_message_size']:
            edge_features = self.edge_feature_resize(edge_features)
            node_features = self.node_feature_resize(node_features)
        edge_features = edge_features.permute(0, 3, 1, 2).to(self.device)
        node_features = node_features.permute(0, 2, 1).to(self.device)

        pred_node_labels = torch.zeros_like(node_labels, device = self.device)
        pred_node_roles = torch.zeros_like(node_roles, device = self.device)
        hidden_node_states = [node_features.clone() for _ in range(self.propagate_layers+1)]
        hidden_edge_states = [edge_features.clone() for _ in range(self.propagate_layers+1)]

        # Belief propagation
        for passing_round in range(self.propagate_layers):
            pred_adj_mat = self.link_fun(hidden_edge_states[passing_round])
            sigmoid_pred_adj_mat = self.sigmoid(pred_adj_mat)

            # Loop through nodes
            for i_node in range(node_features.size()[2]):
                h_v = hidden_node_states[passing_round][:, :, i_node]
                h_w = hidden_node_states[passing_round]
                e_vw = edge_features[:, :, i_node, :]
                m_v = self.message_fun(h_v, h_w, e_vw)

                # Sum up messages from different nodes according to weights
                m_v = sigmoid_pred_adj_mat[:, i_node, :].unsqueeze(1).expand_as(m_v) * m_v
                hidden_edge_states[passing_round+1][:, :, :, i_node] = m_v
                m_v = torch.sum(m_v, 2)
                h_v = self.update_fun(h_v[None].contiguous(), m_v[None])

                # Readout at the final round of message passing
                if passing_round == self.propagate_layers - 1:
                    pred_node_labels[:, i_node, :] = self.readout_fun(h_v.squeeze(0))
                    pred_node_roles[:, i_node, :] = self.readout_fun2(h_v.squeeze(0))

        return pred_adj_mat, pred_node_labels, pred_node_roles

    def _load_link_fun(self, model_args):
        if not os.path.exists(model_args['model_path']):
            os.makedirs(model_args['model_path'])
        best_model_file = os.path.join(model_args['model_path'], os.pardir, 'graph_{}'.format(model_args['feature_type']), 'model_best.pth')
        if os.path.isfile(best_model_file):
            checkpoint = torch.load(best_model_file)
            self.link_fun.load_state_dict(checkpoint['state_dict'])


def main():
    pass


if __name__ == '__main__':
    main()
