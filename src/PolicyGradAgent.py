import torch
import torch.optim as optim
import torch.nn as nn

from src.PolicyGradientNetwork import PolicyNetwork

discount_factor = 0.99
eps = 0.000000001

class PolicyGradAgent:
    def __init__(self, inputSize, outputSize, hiddenSize, lr):
        self.policy_network = PolicyNetwork(inputSize, outputSize, hiddenSize)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def select_action(self, input):
        # Picking from policy
        # with torch.no_grad():
            # softplus = torch.nn.Softplus()

        action_mean_l, action_mean_r = self.policy_network(torch.FloatTensor(input))
        print(action_mean_l, action_mean_r)

        # action_mean = torch.tensor([action_mean_l, action_mean_r])
        # action_std = torch.tensor([std_l, std_r])

        action_dist_l = torch.distributions.Normal(action_mean_l, 2)
        action_dist_r = torch.distributions.Normal(action_mean_r, 2)

        action_l = action_dist_l.sample()
        action_r = action_dist_r.sample()

        action = (action_l.numpy(), action_r.numpy())
        action_prob = action_dist_l.log_prob(action_l) * action_dist_r.log_prob(action_r)

        return action, action_prob.unsqueeze(0)

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        cumulative_reward = 0

        # Process rewards
        for reward in reversed(rewards):
            # Calculate discounted rewards
            cumulative_reward = reward + discount_factor * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate policy loss
        self.optimizer.zero_grad()
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            loss = -abs(-log_prob * reward)
            policy_loss.append(loss)
        policy_loss = torch.cat(policy_loss)
        policy_loss = torch.cat([policy_loss]).sum()

        # Optimization
        policy_loss.backward()
        self.optimizer.step()
        print([x.grad for x in self.policy_network.parameters()])
