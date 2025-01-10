"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch


class UpdateFunction(torch.nn.Module):
    def __init__(self, update_def, args, device):
        super(UpdateFunction, self).__init__()
        self.update_def = update_def.lower()
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
        self.use_transformer = (self.args.get('update_type', 'gru') == 'transformer')
        dropout = 0.0 if dropout == False else dropout

        if not self.use_transformer:
            self.learn_modules.append(torch.nn.GRU(message_size, node_feature_size, num_layers=num_layers, bias=bias, dropout=dropout))
        else:
            embed_dim=self.args.get('embed_dim', 512)
            nhead = self.args.get('transformer_heads', 8)  # Number of attention heads
            dim_feedforward = self.args.get('transformer_feedforward_dim', 2048)
            self.learn_modules.append(torch.nn.Linear(message_size, embed_dim))
            self.learn_modules.append(torch.nn.Linear(node_feature_size, embed_dim))

            # Transformer Decoder (works like GRU update logic)
            transformer = torch.nn.Transformer(
                d_model=embed_dim,
                nhead=nhead,
                num_encoder_layers=num_layers,  # No encoder layers since we use only decoder
                num_decoder_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=False  # Match input shape
            )

            self.learn_modules.append(transformer)
            self.learn_modules.append(torch.nn.Linear(embed_dim,node_feature_size))

    def forward(self, h_v, m_v, args=None):
        if self.use_transformer:
            if h_v.shape[0] == 1:
                h_v = h_v.repeat(m_v.shape[0], 1, 1)

            m_v=self.learn_modules[0](m_v)
            h_v=self.learn_modules[1](h_v)

            # Combine h_v and m_v for Transformer
            transformer_output = self.learn_modules[-2](
                src=m_v,  # Message is the main input (source sequence)
                tgt=h_v  # Node state serves as memory/context
            )
            h = self.learn_modules[-1](transformer_output)
        else:
            output, h = self.learn_modules[0](m_v, h_v)
        return h

def main():
    pass


if __name__ == '__main__':
    main()
