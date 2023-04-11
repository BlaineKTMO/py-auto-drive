import torch
import math
import PyBotSim.src.simulator as Sim
import src.PolicyGradAgent as Agent

class Model:
    def __init__(self, inputSize, outputSize, hiddenSize, learningRate):
        self.sim = Sim.Simulator()
        self.agent = Agent.PolicyGradAgent(inputSize, outputSize,
                                           hiddenSize, learningRate)

    def compute_reward(self, robot_pos, goal_pos):
        distance = ((robot_pos[0] - goal_pos[0]) ** 2 + (robot_pos[1] - goal_pos[1]) ** 2) ** 0.5
        return -distance

    def run_episode(self, goal_position):
        self.sim.reset()
        for _ in range(50):
            input, theta, dist, vl, vr, running = self.sim.step()
        input.append(theta)
        input.append(vl)
        input.append(vr)
        rewards = []
        log_probs = []

        while running:
            action, action_prob = self.agent.select_action(input)
            action_prob.requires_grad = True
            print(action_prob)
            input, theta, dist, vl, vr, running = self.sim.step(action)

            input.append(theta)
            input.append(vl)
            input.append(vr)

            # Compute reward based on progress towards goal
            reward = -abs(dist)
            rewards.append(reward)
            log_probs.append(torch.log(action_prob))
            # print(f"Reward: {reward}")
            # print(f"theta: {theta}")

        # Update policy using accumulated rewards and log probabilities
        self.agent.update_policy(rewards, log_probs)


def main():
    inputSize = 184
    hiddenSize = 250
    outputSize = 2
    learningRate = 0.001

    model = Model(inputSize, outputSize, hiddenSize, learningRate)
    while True:
        model.run_episode((800, 600))
