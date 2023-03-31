import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, laser_scan_size, output_size, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.laser_scan_fc = nn.Linear(laser_scan_size, hidden_size)
        self.heading_fc = nn.Linear(1, hidden_size)
        self.hidden_fc = nn.Linear(2 * hidden_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, laser_scan, heading):
        laser_scan = torch.relu(self.laser_scan_fc(laser_scan))
        heading = torch.relu(self.heading_fc(heading))
        x = torch.cat((laser_scan, heading), dim=1)
        x = torch.relu(self.hidden_fc(x))
        x = self.tanh(self.output_fc(x))
        return x
