"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch
import torch.nn as nn


class ReadoutFunction(nn.Module):
    def __init__(self, readout_def, args, device=None):
        super().__init__()
        self.r_definition = ''
        self.r_function = None
        self.args = {}
        self.learn_args = nn.ParameterList([])
        self.learn_modules = nn.ModuleList([])
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__set_readout(readout_def, args)

    def forward(self, h_v):
        h_v = h_v.to(self.device)
        return self.r_function(h_v)

    # Set a readout function
    def __set_readout(self, readout_def, args):
        self.r_definition = readout_def.lower()
        self.args = args

        self.r_function = {
            'fc': self.r_fc,
            'fc_soft_max': self.r_fc_soft_max,
            'fc_sig': self.r_fc_sigmoid,
        }.get(self.r_definition)

        if self.r_function is None:
            raise ValueError(f"Incorrect definition for readout function: {readout_def}")

        init_parameters = {
            'fc': self.init_fc,
            'fc_soft_max': self.init_fc_soft_max,
            'fc_sig': self.init_fc_sigmoid,
        }.get(self.r_definition, lambda: None)

        init_parameters()

    # Get the name of the used readout function
    def get_definition(self):
        return self.r_definition

    def get_args(self):
        return self.args

    # Definition of readout functions
    def r_fc_soft_max(self, hidden_state):
        last_layer_output = hidden_state
        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        return last_layer_output

    def init_fc_soft_max(self):
        input_size = self.args['readout_input_size']
        output_classes = self.args['output_classes']

        self.learn_modules.append(nn.Linear(input_size, output_classes).to(self.device))
        self.learn_modules.append(nn.Softmax(dim=1).to(self.device))

    def r_fc_sigmoid(self, hidden_state):
        last_layer_output = hidden_state
        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        return last_layer_output

    def init_fc_sigmoid(self):
        input_size = self.args['readout_input_size']
        output_classes = self.args['output_classes']

        self.learn_modules.append(nn.Linear(input_size, input_size).to(self.device))
        self.learn_modules.append(nn.Linear(input_size, output_classes).to(self.device))
        # Uncomment below if sigmoid is needed
        # self.learn_modules.append(nn.Sigmoid().to(self.device))

    def r_fc(self, hidden_state):
        last_layer_output = hidden_state
        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        return last_layer_output

    def init_fc(self):
        input_size = self.args['readout_input_size']
        output_classes = self.args['output_classes']

        self.learn_modules.append(nn.Linear(input_size, input_size).to(self.device))
        self.learn_modules.append(nn.ReLU().to(self.device))
        # Uncomment below if additional layers are needed
        # self.learn_modules.append(nn.Dropout().to(self.device))
        # self.learn_modules.append(nn.BatchNorm1d(input_size).to(self.device))
        self.learn_modules.append(nn.Linear(input_size, output_classes).to(self.device))


def main():
    """
    Unit test for ReadoutFunction
    """
    print("Testing ReadoutFunction on GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test for 'fc' readout function
    args_fc = {
        'readout_input_size': 8,
        'output_classes': 4,
    }
    readout_fc = ReadoutFunction('fc', args_fc, device=device)

    h_v = torch.rand(2, 8).to(device)  # Batch size: 2, Features: 8
    output = readout_fc(h_v)
    assert output.shape == (2, 4)
    print("ReadoutFunction 'fc' test passed!")

    # Test for 'fc_soft_max' readout function
    args_fc_soft_max = {
        'readout_input_size': 8,
        'output_classes': 3,
    }
    readout_fc_soft_max = ReadoutFunction('fc_soft_max', args_fc_soft_max, device=device)

    h_v = torch.rand(3, 8).to(device)  # Batch size: 3, Features: 8
    output = readout_fc_soft_max(h_v)
    assert output.shape == (3, 3)
    print("ReadoutFunction 'fc_soft_max' test passed!")

    # Test for 'fc_sig' readout function
    args_fc_sig = {
        'readout_input_size': 8,
        'output_classes': 2,
    }
    readout_fc_sig = ReadoutFunction('fc_sig', args_fc_sig, device=device)

    h_v = torch.rand(4, 8).to(device)  # Batch size: 4, Features: 8
    output = readout_fc_sig(h_v)
    assert output.shape == (4, 2)
    print("ReadoutFunction 'fc_sig' test passed!")


if __name__ == '__main__':
    main()
