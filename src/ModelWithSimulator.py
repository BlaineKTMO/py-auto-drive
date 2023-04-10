import torch
import PyBotSim.src.simulator as Sim
import src.PolicyGradAgent as Agent

class Model:
    def __init__(self, inputSize, hiddenSize, outputSize, learningRate):
        self.sim = Sim.Simulator()
        self.agent = Agent.PolicyGradAgent(inputSize, hiddenSize,
                                           outputSize, learningRate)

    def compute_reward(self, robot_pos, goal_pos):
        distance = ((robot_pos - goal_pos) ** 2).sum().sqrt()
        return -distance

    def run_episode(self, goal_position):
        self.sim.reset()
        input, _ = self.sim.step()
        done = False
        rewards = []
        log_probs = []

        while not done:
            action, action_prob = self.agent.select_action(input)
            print(action)
            next_input, done = self.sim.step(action)
            self.sim.forward()
            log_probs.append(torch.log(action_prob))

            # Compute reward based on progress towards goal
            current_position = self.sim.robot.getPos()
            reward = self.compute_reward(current_position, goal_position)
            rewards.append(reward)
            log_probs.append(torch.log(action_prob))

        # Update policy using accumulated rewards and log probabilities
        self.agent.update_policy(rewards, log_probs)


def main():
    inputSize = 181
    hiddenSize = 100
    outputSize = 2
    learningRate = 0.01

    model = Model(inputSize, hiddenSize, outputSize, learningRate)
    model.run_episode((800, 800))
