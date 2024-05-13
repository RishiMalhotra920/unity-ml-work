from a2c.neural_net import NeuralNet, PiNetwork
import torch

# NUM_ACTIONS = 32*32

pi_network = PiNetwork(8, 20, 20, 20, 2) # [s] -> a
v_network = NeuralNet(8, 20, 20, 1)


def load_pi_network(file_path=None):
  pi_network.load_state_dict(torch.load(file_path))
  return pi_network

