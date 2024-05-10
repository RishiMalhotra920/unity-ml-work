import numpy as np
import torch 

class RolloutBuffer:
  def __init__(self, buffer_size, state_dim, action_dim):
    self.buffer_size = buffer_size
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.reset()
  
  def add(self, state, action, reward, state_prime):
    # print('array of state', np.array(state))
    self.states[self.pos] = np.array(state)
    # print('this is action: may have to fix the form', action)
    self.actions[self.pos] = np.array(action)
    # print('self.actions', self.actions)
    self.rewards[self.pos] = reward
    self.s_primes[self.pos] = np.array(state_prime)
    self.pos += 1
    
  def set_end(self, end):
    self.end = end
  
  def get_end(self):
    if self.end == None:
      raise Exception("end not set")
    
    return self.end
  
  def get_last_state(self):
    return self.s_primes[-1]
    # return self.rollout[-1][3]
  
  def reset(self):
    self.states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
    self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
    self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
    self.s_primes = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
    self.returns = np.zeros((self.buffer_size), dtype=np.float32)
    self.end = None
    self.pos = 0
  
  def compute_returns(self, value_network, gamma):
    G = 0
    if self.end:
      G = 0
    else:
      s_last = self.get_last_state()
      G = value_network(torch.tensor(s_last)).detach().item()

    for i in range(len(self.states)-1, -1, -1):
      G = self.rewards[i] + gamma * G
      self.returns[i] = G


  def get_rollouts(self):
    return self.states[:self.pos], self.actions[:self.pos], self.rewards[:self.pos], self.s_primes[:self.pos], self.returns[:self.pos], self.end
  
  def get_rollout_tensors(self):
    return (
      torch.tensor(self.states[:self.pos]),
      torch.tensor(self.actions[:self.pos]),
      torch.tensor(self.rewards[:self.pos]),
      torch.tensor(self.s_primes[:self.pos]),
      torch.tensor(self.returns[:self.pos]),
      self.end
    )
  
  def __len__(self):
    """
    return num steps taken in rollout and not the buffer size
    """
    return self.pos
  
  def __getitem__(self, key):
    return (
      self.states[key],
      self.actions[key],
      self.rewards[key],
      self.s_primes[key],
      self.returns[key],
      self.end
    )