# from neural_net import NeuralNet
import torch
import torch.nn as nn
import numpy as np


class PiNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, hidden_size2, hidden_size3, num_means):
    super(PiNetwork, self).__init__()

    self.l1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size, hidden_size2)
    self.l3 = nn.Linear(hidden_size2, hidden_size3)
    self.mu_head = nn.Linear(hidden_size3, num_means)
    self.sigma_head = nn.Linear(hidden_size3, num_means)
    
  
  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    out = self.relu(out)
    out = self.l3(out)
    out = self.relu(out)

    mu_out = self.mu_head(out)
    sigma_out = self.sigma_head(out)

    return mu_out, sigma_out

class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, hidden_size2, num_classes):
    super(NeuralNet, self).__init__()

    self.l1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size, hidden_size2)
    self.l3 = nn.Linear(hidden_size2, num_classes)
    
  
  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    out = self.relu(out)
    out = self.l3(out)
    return out
