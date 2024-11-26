"""
Created on Oct 07, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import torch
import torch.nn as nn
import units


class GPNN_HICO(nn.Module):
    def __init__(self, model_args, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_args = model_args.copy()

        # Resize features if needed
        if model_args['resize_feature_to_message_size']:
            self.edge_feature_resize = nn.Linear(model_args['edge_feature_size'], model_args['message_size']).to(self.device)
            self.node_feature_resize = nn.Linear(model_args['node_feature_size'], model_args['message_size']).to(self.device)
            nn.init.xavier_normal_(self.edge_feature_resize.weight)
            nn.init.xavier_normal_(self.node_feature_resize.weight)

            model_args['edge_feature_size'] = model_args['message_size']
            model_args['node_feature_size'] = model_args['message_size']

        # Initialize functions
        self.link_fun = units.LinkFunction('GraphConv', model_args).to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)
        self.message_fun = units.MessageFunction('linear_concat_relu', model_args).to(self.device)
        self.update_fun = units.UpdateFunction('gru', model_args).to(self.device)
        self.readout_fun = units.ReadoutFunction(
            'fc', {'readout_input_size': model_args['node_feature_size'], 'output_classes': model_args['hoi_classes']}
        ).to(self.device)

        self.propagate_layers = model_args['propagate_layers']
        self._load_link_fun(model_args)

    def forward(self, edge_features, node_features, adj_mat, node_labels, human_nums, obj_nums, args):
        # Resize features if necessary
        if self.model_args['resize_feature_to_message_size']:
            edge_features = self.edge_feature_resize(edge_features)
            node_features = self.node_feature_resize(node_features)
        edge_features = edge_features.permute(0, 3, 1, 2).to(self.device)
        node_features = node_features.permute(0, 2, 1).to(self.device)

        batch_size, num_nodes = node_features.size(0), node_features.size(2)

        # Initialize hidden states and outputs
        pred_adj_mat = torch.zeros_like(adj_mat, device=self.device)
        pred_node_labels = torch.zeros_like(node_labels, device=self.device)
        hidden_node_states = [[node_features[batch_idx].unsqueeze(0).clone() for _ in range(self.propagate_layers + 1)]
                              for batch_idx in range(batch_size)]
        hidden_edge_states = [[edge_features[batch_idx].unsqueeze(0).clone() for _ in range(self.propagate_layers + 1)]
                              for batch_idx in range(batch_size)]

        # Belief propagation
        for batch_idx in range(batch_size):
            valid_node_num = human_nums[batch_idx] + obj_nums[batch_idx]
            for passing_round in range(self.propagate_layers):
                # Predict adjacency matrix
                pred_adj_mat[batch_idx, :valid_node_num, :valid_node_num] = self.link_fun(
                    hidden_edge_states[batch_idx][passing_round][:, :, :valid_node_num, :valid_node_num]
                )
                sigmoid_pred_adj_mat = self.sigmoid(pred_adj_mat[batch_idx, :valid_node_num, :valid_node_num]).unsqueeze(0)

                # Process each node
                
                for i_node in range(valid_node_num):
                   
                    
                    h_v = hidden_node_states[batch_idx][passing_round][:, :, i_node]
                    
                    h_w = hidden_node_states[batch_idx][passing_round][:, :, :valid_node_num]
                    e_vw = edge_features[batch_idx, :, i_node, :valid_node_num].unsqueeze(0)

                    m_v = self.message_fun(h_v, h_w, e_vw, args)

                    # Aggregate messages
                    m_v = sigmoid_pred_adj_mat[:, i_node, :valid_node_num].unsqueeze(1).expand_as(m_v) * m_v
                    hidden_edge_states[batch_idx][passing_round + 1][:, :, :valid_node_num, i_node] = m_v
                    m_v = torch.sum(m_v, dim=2)
                    h_v = self.update_fun(h_v[None].contiguous(), m_v[None])

                    # Readout at final round
                    if passing_round == self.propagate_layers - 1:
                        pred_node_labels[batch_idx, i_node, :] = self.readout_fun(h_v.squeeze(0))

        return pred_adj_mat, pred_node_labels

    def _load_link_fun(self, model_args):
        if not os.path.exists(model_args['model_path']):
            os.makedirs(model_args['model_path'])
        best_model_file = os.path.join(model_args['model_path'], os.pardir, 'graph', 'model_best.pth')
        if os.path.isfile(best_model_file):
            checkpoint = torch.load(best_model_file, map_location=self.device)
            self.link_fun.load_state_dict(checkpoint['state_dict'])


def main():
    """
    Unit test for GPNN_HICO
    """
    print("Testing GPNN_HICO on GPU...")

    # Define test model_args
    model_args = {
        'node_feature_size': 16,
        'edge_feature_size': 6,
        'hoi_classes': 10,
        'message_size': 6,
        'resize_feature_to_message_size': False,
        'propagate_layers': 2,
        'link_hidden_size':3,
        'link_hidden_layers':2,
        'model_path': './model_checkpoints/',
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test inputs
    batch_size = 3
    num_nodes = 10
    edge_features = torch.rand(batch_size, num_nodes, num_nodes, 6).to(device)
    node_features = torch.rand(batch_size, num_nodes, 16).to(device)
    adj_mat = torch.rand(batch_size, num_nodes, num_nodes).to(device)
    node_labels = torch.rand(batch_size, num_nodes, 10).to(device)

    """Human nums + obj nums <= num nodes, they all are nodes but are not all nodes, but valid."""
    human_nums = torch.randint(1, 6, (batch_size,)).to(device)
    obj_nums = torch.full((batch_size,), num_nodes).to(device)-human_nums
   
    # Initialize model
    model = GPNN_HICO(model_args, device=device)
    pred_adj_mat, pred_node_labels = model(
        edge_features, node_features, adj_mat, node_labels, human_nums, obj_nums, args={'cuda': True}
    )

    # Assertions
    assert pred_adj_mat.shape == adj_mat.shape
    assert pred_node_labels.shape == node_labels.shape
    print("All tests passed!")


if __name__ == '__main__':
    main()
