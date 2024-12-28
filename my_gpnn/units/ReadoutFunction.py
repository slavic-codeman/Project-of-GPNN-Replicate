"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch


class ReadoutFunction(torch.nn.Module):
    def __init__(self, readout_def, args, device):
        super(ReadoutFunction, self).__init__()
        self.readout_def = readout_def
        self.args = args
        self.device = device

        self.sigmoid = 'sig' in self.readout_def 
        self.softmax = 'soft_max' in self.readout_def
        self.learn_modules = torch.nn.ModuleList()
        self.init_fc()

    # Get the name of the used readout function
    def get_definition(self):
        return self.readout_def

    def get_args(self):
        return self.args
    
    def init_fc(self):
        input_size = self.args['readout_input_size']
        output_classes = self.args['output_classes']
        # 此处和源码结构不同
        self.learn_modules.append(torch.nn.Linear(input_size, input_size))
        self.learn_modules.append(torch.nn.ReLU())
        self.learn_modules.append(torch.nn.Linear(input_size, output_classes))
        if self.sigmoid:
            self.learn_modules.append(torch.nn.Sigmoid())
        if self.softmax:
            self.learn_modules.append(torch.nn.Softmax())
    
    def forward(self, h_v):
        last_layer_output = h_v
        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        return last_layer_output

    # Fully connected layers with softmax output
    def r_fc_sigmoid(self, hidden_state):
        last_layer_output = hidden_state
        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        return last_layer_output


def main():
    pass


if __name__ == '__main__':
    main()
