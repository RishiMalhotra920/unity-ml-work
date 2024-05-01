from a2c.neural_net import NeuralNet
import torch


actor = NeuralNet(8, 20, 20, 2) #s->a
critic = NeuralNet(10, 20, 20, 1) # [s, a] -> r


def load_critic(critic_file_path=None):
  critic.load_state_dict(torch.load(critic_file_path))
  return critic

def load_actor(actor_file_path=None):
  actor.load_state_dict(torch.load(actor_file_path))
  return actor