import os

import torch
import torch.nn
import torch.autograd

import units


class GPNN_CAD(torch.nn.Module):
    def __init__(self, model_args, device = None):
        super(GPNN_CAD, self).__init__()

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.link_fun = units.LinkFunction('GraphConvLSTM', model_args, self.device).to(self.device)
        self.message_fun = units.MessageFunction('linear_concat', model_args, self.device).to(self.device)

        self.update_funs = torch.nn.ModuleList([
        units.UpdateFunction('transformer' if model_args.get('transformer', False) else 'gru', model_args, self.device).to(self.device),
        units.UpdateFunction('transformer' if model_args.get('transformer', False) else 'gru', model_args, self.device).to(self.device)
        ])

        self.subactivity_classes = model_args['subactivity_classes']
        self.affordance_classes = model_args['affordance_classes']
        self.readout_funs = torch.nn.ModuleList([
        units.ReadoutFunction('fc_soft_max', {'readout_input_size': model_args['node_feature_size'], 'output_classes': self.subactivity_classes}, self.device).to(self.device),
        units.ReadoutFunction('fc_soft_max', {'readout_input_size': model_args['node_feature_size'], 'output_classes': self.affordance_classes}, self.device).to(self.device)
        ])

        self.propagate_layers = model_args['propagate_layers']

        self._load_link_fun(model_args)

    def forward(self, edge_features, node_features, adj_mat, node_labels, args):
        pred_node_labels = torch.zeros(node_labels.size(), device = self.device)

        hidden_node_states = [node_features.clone() for _ in range(self.propagate_layers+1)]
        hidden_edge_states = [edge_features.clone() for _ in range(self.propagate_layers+1)]

        # Belief propagation
        for passing_round in range(self.propagate_layers):
            pred_adj_mat = self.link_fun(hidden_edge_states[passing_round])

            # Loop through nodes
            for i_node in range(node_features.size()[2]):
                h_v = hidden_node_states[passing_round][:, :, i_node]
                h_w = hidden_node_states[passing_round]
                e_vw = edge_features[:, :, i_node, :]
                m_v = self.message_fun(h_v, h_w, e_vw)

                # Sum up messages from different nodes according to weights
                m_v = pred_adj_mat[:, i_node, :].unsqueeze(1).expand_as(m_v) * m_v
                hidden_edge_states[passing_round+1][:, :, :, i_node] = m_v

                m_v = torch.sum(m_v, 2)

                h_v = self.update_funs[0 if i_node == 0 else 1](h_v[None].contiguous(), m_v[None])

                # Readout at the final round of message passing
                if passing_round == self.propagate_layers-1:
                    if i_node == 0:
                        pred_node_labels[:, i_node, :self.subactivity_classes] = self.readout_funs[0](h_v.squeeze(0))
                    else:
                        pred_node_labels[:, i_node, :] = self.readout_funs[1](h_v.squeeze(0))

        return pred_adj_mat, pred_node_labels

    def _load_link_fun(self, model_args):
        if not os.path.exists(model_args['model_path']):
            os.makedirs(model_args['model_path'])
        best_model_file = os.path.join(model_args['model_path'], '..', 'graph', 'model_best.pth')
        if os.path.isfile(best_model_file):
            checkpoint = torch.load(best_model_file)
            self.link_fun.load_state_dict(checkpoint['state_dict'])

    def _dump_link_fun(self, model_args):
        if not os.path.exists(model_args['model_path']):
            os.makedirs(model_args['model_path'])
        if not os.path.exists(os.path.join(model_args['model_path'], '..', 'graph')):
            os.makedirs(os.path.join(model_args['model_path'], '..', 'graph'))
        best_model_file = os.path.join(model_args['model_path'], '..', 'graph', 'model_best.pth')
        torch.save({'state_dict': self.link_fun.state_dict()}, best_model_file)


def main():
    pass


if __name__ == '__main__':
    main()
