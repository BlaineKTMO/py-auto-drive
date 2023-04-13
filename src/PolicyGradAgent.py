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
        #     action = self.policy_network(torch.FloatTensor(input))
        #     action_dist = torch.distributions.Categorical(action_probs)
        #     action = action_dist.sample()
        # return action.item(), action_probs[action]
        with torch.no_grad():
            action_mean_l, action_mean_r, d1, d2, d3 = self.policy_network(torch.FloatTensor(input))
            action_mean = torch.tensor([action_mean_l, action_mean_r], requires_grad=True)
            softplus = torch.nn.Softplus()
            d1 = softplus(d1)
            d2 = softplus(d2)
            d3 = softplus(d3)
            d = torch.tensor([d1, d2, d3])
            cov = torch.zeros(2, 2)
            cov[torch.tril_indices(2, 2, offset=0).tolist()] = d
            cov = cov * torch.transpose(cov, 0, 1)
            action_dist = torch.distributions.MultivariateNormal(action_mean, cov + eps)
            action = action_dist.sample()
            # print(action_dist.log_prob(action))
        return action.numpy(), action_dist.log_prob(action).unsqueeze(0)

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

        policy_loss = []
        print(log_probs)
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        print(policy_loss)
        policy_loss = torch.cat(policy_loss).sum()

        # Optimization
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
