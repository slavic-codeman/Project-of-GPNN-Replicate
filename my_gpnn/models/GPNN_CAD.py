"""
Created on Oct 07, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import torch
import torch.nn as nn
import units


class GPNN_CAD(nn.Module):
    def __init__(self, model_args, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize link, message, update, and readout functions
        self.link_fun = units.LinkFunction('GraphConvLSTM', model_args).to(self.device)
        self.message_fun = units.MessageFunction('linear_concat', model_args).to(self.device)

        self.update_funs = nn.ModuleList([
            units.UpdateFunction('gru', model_args).to(self.device),
            units.UpdateFunction('gru', model_args).to(self.device)
        ])

        self.subactivity_classes = model_args['subactivity_classes']
        self.affordance_classes = model_args['affordance_classes']

        self.readout_funs = nn.ModuleList([
            units.ReadoutFunction('fc_soft_max', {
                'readout_input_size': model_args['node_feature_size'],
                'output_classes': self.subactivity_classes
            }).to(self.device),
            units.ReadoutFunction('fc_soft_max', {
                'readout_input_size': model_args['node_feature_size'],
                'output_classes': self.affordance_classes
            }).to(self.device)
        ])

        self.propagate_layers = model_args['propagate_layers']

        self._load_link_fun(model_args)

    def forward(self, edge_features, node_features, adj_mat, node_labels, args):
        pred_node_labels = torch.zeros(node_labels.size(), device=self.device)
        hidden_node_states = [node_features for _ in range(self.propagate_layers + 1)]
        
        hidden_edge_states = [edge_features.clone() for _ in range(self.propagate_layers + 1)]

        # Belief propagation
        for passing_round in range(self.propagate_layers):
            pred_adj_mat = self.link_fun(hidden_edge_states[passing_round])

            for i_node in range(node_features.size(2)):
                h_v = hidden_node_states[passing_round][:, :, i_node]
                h_w = hidden_node_states[passing_round]
                e_vw = edge_features[:, :, i_node, :]

                m_v = self.message_fun(h_v, h_w, e_vw, args)
                
                # Sum up messages from different nodes according to weights
                m_v = pred_adj_mat[:, i_node, :].unsqueeze(1).expand_as(m_v) * m_v
                hidden_edge_states[passing_round + 1][:, :, :, i_node] = m_v
         
                m_v = torch.sum(m_v, dim=2)

                if i_node == 0:
                    h_v = self.update_funs[0](h_v[None].contiguous(), m_v[None])
                else:
                    h_v = self.update_funs[1](h_v[None].contiguous(), m_v[None])

                if passing_round == self.propagate_layers - 1:
                    if i_node == 0:
                        pred_node_labels[:, i_node, :self.subactivity_classes] = self.readout_funs[0](h_v.squeeze(0))
                    else:
                        # TODO Modified shape? concatenation or inclusion of subactivity in affordance?
                        pred_node_labels[:, i_node, :] = self.readout_funs[1](h_v.squeeze(0))
                      
                        #pred_node_labels[:, i_node, self.subactivity_classes:] = self.readout_funs[1](h_v.squeeze(0))

        return pred_adj_mat, pred_node_labels

    def _load_link_fun(self, model_args):
        if not os.path.exists(model_args['model_path']):
            os.makedirs(model_args['model_path'])
        best_model_file = os.path.join(model_args['model_path'], '..', 'graph', 'model_best.pth')
        if os.path.isfile(best_model_file):
            checkpoint = torch.load(best_model_file, map_location=self.device)
            self.link_fun.load_state_dict(checkpoint['state_dict'])

    def _dump_link_fun(self, model_args):
        if not os.path.exists(model_args['model_path']):
            os.makedirs(model_args['model_path'])
        if not os.path.exists(os.path.join(model_args['model_path'], '..', 'graph')):
            os.makedirs(os.path.join(model_args['model_path'], '..', 'graph'))
        best_model_file = os.path.join(model_args['model_path'], '..', 'graph', 'model_best.pth')
        torch.save({'state_dict': self.link_fun.state_dict()}, best_model_file)


def main():
    """
    Unit test for GPNN_CAD
    """
    print("Testing GPNN_CAD on GPU...")

    # Define test model_args
    model_args = {
        'node_feature_size':2,
        'edge_feature_size': 4,
        'subactivity_classes': 10,
        'affordance_classes': 12,
        'propagate_layers': 6,
        'link_hidden_size': 7,
        'link_hidden_layers': 8,
        'model_path': './model_checkpoints/',
        
        'message_size': 4,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test inputs
    batch_size = 10
    num_nodes = 11
    edge_features = torch.rand(batch_size, 4, num_nodes, num_nodes).to(device)
    node_features = torch.rand(batch_size, 2, num_nodes).to(device)
    adj_mat = torch.rand(batch_size, num_nodes, num_nodes).to(device)
    node_labels = torch.rand(batch_size, num_nodes,model_args['affordance_classes']).to(device)

    # Initialize model
    model = GPNN_CAD(model_args, device=device)
    pred_adj_mat, pred_node_labels = model(edge_features, node_features, adj_mat, node_labels, args={'cuda': True})


    a=torch.mean(pred_adj_mat)
    a.backward()
    assert pred_adj_mat.shape == adj_mat.shape
    assert pred_node_labels.shape == node_labels.shape
    print("All tests passed!")


if __name__ == '__main__':
    main()
