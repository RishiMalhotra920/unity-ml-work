from a2c.neural_net import NeuralNet
import torch



q_network = NeuralNet(10, 20, 20, 1) # [s, a] -> r


def load_q_network(file_path=None):
  q_network.load_state_dict(torch.load(file_path))
  return q_network