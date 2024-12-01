import torch
import torch.nn as nn


class UpdateFunction(nn.Module):
    def __init__(self, update_def, args, device=None):
        super().__init__()
        self.u_definition = ''
        self.u_function = None
        self.args = {}
        self.learn_args = nn.ParameterList([])
        self.learn_modules = nn.ModuleList([])
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__set_update(update_def, args)

    def forward(self, h_v, m_v, args=None):
        h_v, m_v = h_v.to(self.device), m_v.to(self.device)
       
        return self.u_function(h_v, m_v, args)

    # Set an update function
    def __set_update(self, update_def, args):
        self.u_definition = update_def.lower()
        self.args = args

        self.u_function = {
            'gru': self.u_gru,
        }.get(self.u_definition)

        if self.u_function is None:
            raise ValueError(f"Incorrect update definition: {update_def}")

        init_parameters = {
            'gru': self.init_gru,
        }.get(self.u_definition, lambda: None)

        init_parameters()

    # Get the name of the used update function
    def get_definition(self):
        return self.u_definition

    def get_args(self):
        return self.args

    # Definition of update functions
    # GRU: node state as hidden state, message as input
    def u_gru(self, h_v, m_v, args):
     
        output, h = self.learn_modules[0](m_v, h_v)
        return h

    def init_gru(self):
        node_feature_size = self.args['node_feature_size']
        message_size = self.args['message_size']
        num_layers = self.args.get('update_hidden_layers', 1)
        bias = self.args.get('update_bias', False)
        dropout = self.args.get('update_dropout', 0.0) if num_layers > 1 else 0.0
        self.learn_modules.append(
            nn.GRU(
                input_size=message_size,
                hidden_size=node_feature_size,
                num_layers=num_layers,
                bias=bias,
                dropout=dropout,
                batch_first=False
            ).to(self.device)
        )


def main():
    """
    Unit test for UpdateFunction
    """
    # Test parameters
    args = {
        'node_feature_size': 16,
        'message_size': 8,
        'update_hidden_layers': 1,
        'update_bias': True,
        'update_dropout': 0.1,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    seq_length = 12

    # Randomly generate GRU input tensors
    h_v = torch.randn(args['update_hidden_layers'], batch_size, args['node_feature_size']).to(device)  # Hidden state
    m_v = torch.randn( seq_length,batch_size, args['message_size']).to(device)  # Input message

    # Test GRU mode
    print("Testing GRU mode on GPU...")
    update_function = UpdateFunction('gru', args, device=device)
    output = update_function(h_v, m_v)
    assert output.shape == h_v.shape  # Ensure the output shape matches the hidden state
    print(f"GRU output shape: {output.shape}")
    print("All tests passed!")


if __name__ == '__main__':
    main()
