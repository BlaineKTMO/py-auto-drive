import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, input, output_size, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.input_fc = nn.Linear(1, hidden_size)
        self.hidden_fc = nn.Linear(hidden_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = torch.input_fc(input)
        output = torch.relu(self.hidden_fc(output))
        output = self.output_fc(output)
        output = self.tanh(output)
        return output
