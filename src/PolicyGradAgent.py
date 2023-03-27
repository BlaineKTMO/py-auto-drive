import torch
import torch.optim as optim
import numpy as np

from src.PolicyGradientNetwork import PolicyNetwork


class PolicyGradAgent:
    def __init__(self, input_size, output_size, hidden_size, lr):
        self.policy_network = PolicyNetwork(input_size, output_size, hidden_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.policy_network(state)
        action_probs = action_probs.detach().numpy().squeeze()
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action, np.log(action_probs[action])

    def update_policy(self, rewards, log_probs):
        returns = [0] * len(rewards)
        G = 0

        for t in range(len(rewards) - 1, -1, -1):
            G = rewards[t] + 0.99 * G
            returns[t] = G

        returns = torch.tensor(returns)
        log_probs = torch.stack(log_probs)
        loss = -log_probs * returns
        loss = torch.sum(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
