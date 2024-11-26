"""
Created on FEB 25, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import torch
import torch.nn as nn
import units


class GPNN_VCOCO(nn.Module):
    def __init__(self, model_args, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_args = model_args.copy()
        if model_args['resize_feature_to_message_size']:
            # Resize large features
            self.edge_feature_resize = nn.Linear(model_args['edge_feature_size'], model_args['message_size']).to(self.device)
            self.node_feature_resize = nn.Linear(model_args['node_feature_size'], model_args['message_size']).to(self.device)
            nn.init.xavier_normal_(self.edge_feature_resize.weight)
            nn.init.xavier_normal_(self.node_feature_resize.weight)

            model_args['edge_feature_size'] = model_args['message_size']
            model_args['node_feature_size'] = model_args['message_size']

        self.link_fun = units.LinkFunction('GraphConv', model_args).to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)
        self.message_fun = units.MessageFunction('linear_concat_relu', model_args).to(self.device)
        self.update_fun = units.UpdateFunction('gru', model_args).to(self.device)
        self.readout_fun = units.ReadoutFunction(
            'fc', {'readout_input_size': model_args['node_feature_size'], 'output_classes': model_args['hoi_classes']}
        ).to(self.device)
        self.readout_fun2 = units.ReadoutFunction(
            'fc', {'readout_input_size': model_args['node_feature_size'], 'output_classes': model_args['roles_num']}
        ).to(self.device)

        self.propagate_layers = model_args['propagate_layers']

        self._load_link_fun(model_args)

    def forward(self, edge_features, node_features, adj_mat, node_labels, node_roles, human_nums, obj_nums, args):
        if self.model_args['resize_feature_to_message_size']:
            edge_features = self.edge_feature_resize(edge_features)
            node_features = self.node_feature_resize(node_features)
        edge_features = edge_features.permute(0, 3, 1, 2).to(self.device)
        node_features = node_features.permute(0, 2, 1).to(self.device)

        pred_node_labels = torch.zeros_like(node_labels, device=self.device)
        pred_node_roles = torch.zeros_like(node_roles, device=self.device)
        hidden_node_states = [node_features.clone() for _ in range(self.propagate_layers + 1)]
        hidden_edge_states = [edge_features.clone() for _ in range(self.propagate_layers + 1)]

        # Belief propagation
        for passing_round in range(self.propagate_layers):
            pred_adj_mat = self.link_fun(hidden_edge_states[passing_round])
            sigmoid_pred_adj_mat = self.sigmoid(pred_adj_mat)

            for i_node in range(node_features.size(2)):
                h_v = hidden_node_states[passing_round][:, :, i_node]
                h_w = hidden_node_states[passing_round]
                e_vw = edge_features[:, :, i_node, :]

                m_v = self.message_fun(h_v, h_w, e_vw, args)

                # Sum up messages from different nodes according to weights
                m_v = sigmoid_pred_adj_mat[:, i_node, :].unsqueeze(1).expand_as(m_v) * m_v
                hidden_edge_states[passing_round + 1][:, :, :, i_node] = m_v
                m_v = torch.sum(m_v, dim=2)
                h_v = self.update_fun(h_v[None].contiguous(), m_v[None])

                if passing_round == self.propagate_layers - 1:
                    pred_node_labels[:, i_node, :] = self.readout_fun(h_v.squeeze(0))
                    pred_node_roles[:, i_node, :] = self.readout_fun2(h_v.squeeze(0))

        return pred_adj_mat, pred_node_labels, pred_node_roles

    def _load_link_fun(self, model_args):
        if not os.path.exists(model_args['model_path']):
            os.makedirs(model_args['model_path'])
        best_model_file = os.path.join(
            model_args['model_path'], os.pardir, f"graph_{model_args['feature_type']}", 'model_best.pth'
        )
        if os.path.isfile(best_model_file):
            checkpoint = torch.load(best_model_file, map_location=self.device)
            self.link_fun.load_state_dict(checkpoint['state_dict'])


def main():
    """
    Unit test for GPNN_VCOCO
    """
    print("Testing GPNN_VCOCO on GPU...")

    # Define test model_args
    model_args = {
        'node_feature_size': 2,
        'edge_feature_size': 4,
        'subactivity_classes': 4,
        'affordance_classes': 5,
        'propagate_layers': 6,
        'link_hidden_size': 7,
        'link_hidden_layers': 8,
        'update_hidden_layers': 1, # only 1, since extract 1 hidden each time
        'model_path': './model_checkpoints/',
        'resize_feature_to_message_size':False,
        'message_size': 4, # = edge feature size, and %2==0
        "hoi_classes": 11,
        "roles_num":12,
        "feature_type":13
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test inputs
    batch_size = 12
    num_nodes =15
    edge_features = torch.rand(batch_size, num_nodes, num_nodes, 4).to(device)
    node_features = torch.rand(batch_size, num_nodes, 2).to(device)
    adj_mat = torch.rand(batch_size, num_nodes, num_nodes).to(device)
    node_labels = torch.rand(batch_size, num_nodes, 11).to(device)
    node_roles = torch.rand(batch_size, num_nodes, 12).to(device)
    human_nums = torch.randint(1, num_nodes + 1, (batch_size,)).to(device)
    obj_nums = torch.randint(1, num_nodes + 1, (batch_size,)).to(device)

    # Initialize model
    model = GPNN_VCOCO(model_args, device=device)
    pred_adj_mat, pred_node_labels, pred_node_roles = model(
        edge_features, node_features, adj_mat, node_labels, node_roles, human_nums, obj_nums, args={'cuda': True}
    )

    assert pred_adj_mat.shape == adj_mat.shape
    assert pred_node_labels.shape == node_labels.shape
    assert pred_node_roles.shape == node_roles.shape
    print("All tests passed!")


if __name__ == '__main__':
    main()
