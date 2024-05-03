from a2c.neural_net import NeuralNet
import torch

# NUM_ACTIONS = 32*32

pi_network = NeuralNet(8, 20, 20, 4) # [s] -> a


def load_pi_network(file_path=None):
  pi_network.load_state_dict(torch.load(file_path))
  return pi_network

