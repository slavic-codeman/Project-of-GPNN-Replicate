import sys
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, kernel_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, self.kernel_size, padding=self.padding)

    def forward(self, input_, prev_state=None, use_cuda=True):
        # Get batch and spatial sizes
        batch_size, _, height, width = input_.size()

        # Generate empty prev_state if None is provided
        state_size = (batch_size, self.hidden_size, height, width)
        if prev_state is None or prev_state[0].size() != input_.size():
            prev_state = self._reset_prev_states(state_size, use_cuda)

        prev_hidden, prev_cell = prev_state

        # Data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), dim=1)
        gates = self.Gates(stacked_inputs)

        # Chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, dim=1)

        # Apply nonlinearities
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        # Compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

    @staticmethod
    def _reset_prev_states(state_size, use_cuda):
        device = torch.device("cuda" if use_cuda else "cpu")
        return (
            torch.zeros(state_size, device=device),
            torch.zeros(state_size, device=device),
        )


class ConvLSTM(nn.Module):
    """
    ConvLSTM module that supports multiple stacked ConvLSTM layers
    """

    def __init__(self, input_channels, hidden_channels, hidden_layer_num, kernel_size=1, bias=True):
        super().__init__()
        if hidden_layer_num < 1:
            raise ValueError("Hidden layer number must be at least 1.")

        self.hidden_layer_num = hidden_layer_num
        self.learn_modeles = nn.ModuleList()
        self.prev_states = [None] * hidden_layer_num

        # First layer
        self.learn_modeles.append(ConvLSTMCell(input_channels, hidden_channels, kernel_size))
        # Subsequent layers
        for _ in range(hidden_layer_num - 1):
            self.learn_modeles.append(ConvLSTMCell(hidden_channels, hidden_channels, kernel_size))

    def forward(self, input_, reset=False):
        if reset:
            self._reset_hidden_states()
        else:
            # TODO: modified how to updata prev states
            for i in range(len(self.prev_states)):
                if self.prev_states[i] is not None:
                    self.prev_states[i] = (
                        self.prev_states[i][0].clone().detach(),
                        self.prev_states[i][1].clone().detach()
                    )

        next_layer_input = input_
        for i, layer in enumerate(self.learn_modeles):
            prev_state = layer(next_layer_input, self.prev_states[i])
            next_layer_input = prev_state[0]
            self.prev_states[i] = prev_state
            
        return next_layer_input

    def _reset_hidden_states(self):
        self.prev_states = [None for _ in range(self.hidden_layer_num)]


def main():
    """
    Unit test for ConvLSTMCell and ConvLSTM
    """
    # Testing ConvLSTMCell
    device=torch.device("cuda")
    print("Testing ConvLSTMCell...")
    input_size = 3
    hidden_size = 5
    kernel_size = 3
    cell = ConvLSTMCell(input_size, hidden_size, kernel_size).to(device)

    batch_size = 2
    height, width = 8, 8
    input_tensor = torch.rand(batch_size, input_size, height, width).to(device)
    prev_hidden = torch.zeros(batch_size, hidden_size, height, width).to(device)
    prev_cell = torch.zeros(batch_size, hidden_size, height, width).to(device)

    hidden, cell_output = cell(input_tensor, (prev_hidden, prev_cell))
    assert hidden.shape == (batch_size, hidden_size, height, width)
    assert cell_output.shape == (batch_size, hidden_size, height, width)
    print("ConvLSTMCell forward pass test passed.")

    # Testing ConvLSTM
    print("Testing ConvLSTM...")
    input_channels = 3
    hidden_channels = 5
    hidden_layer_num = 2
    lstm = ConvLSTM(input_channels, hidden_channels, hidden_layer_num, kernel_size).to(device)


    for t in range(2):
        lstm._reset_hidden_states()
        seq_len = 4
        inputs = torch.rand(seq_len, batch_size, input_channels, height, width).to(device)
        outputs = []
        for t in range(seq_len):
            output = lstm(inputs[t])
            outputs.append(output)
        a=torch.mean(outputs[-1])
        a.backward()
    assert outputs[-1].shape == (batch_size, hidden_channels, height, width)
    print("ConvLSTM forward pass test passed.")


if __name__ == "__main__":
    main()
