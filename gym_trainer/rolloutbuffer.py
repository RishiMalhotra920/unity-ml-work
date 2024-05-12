import numpy as np
import torch 

class Rollouts:
  def __init__(self, num_rollouts, buffer_size, state_dim, action_dim):
    self.num_rollouts = num_rollouts
    self.buffer_size = buffer_size
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.rollouts = [RolloutBuffer(buffer_size, state_dim, action_dim) for _ in range(num_rollouts)]
  
  def add(self, rollout_idx, state, action, reward, state_prime):
    self.rollouts[rollout_idx].add(state, action, reward, state_prime)
  
  def set_end(self, rollout_idx, end):
    self.rollouts[rollout_idx].set_end(end)
  
  def get_rollout_tensors(self):
    states = []
    actions = []
    rewards = []
    s_primes = []
    returns = []
    ends = []
    for r in self.rollouts:
      s, a, r, s_prime, ret, end = r.get_rollout_tensors()
      states.append(s)
      actions.append(a)
      rewards.append(r)
      s_primes.append(s_prime)
      returns.append(ret)
      ends.append(end)
    return (
      torch.cat(states),
      torch.cat(actions),
      torch.cat(rewards),
      torch.cat(s_primes),
      torch.cat(returns),
      ends
    )

  def compute_returns(self, value_network, gamma):
    for r in self.rollouts:
      r.compute_returns(value_network, gamma)
  
  def reset(self):
    for r in self.rollouts:
      r.reset()
  
  def write(self):
    with open('rollouts.txt', 'w') as f:
      for i, r in enumerate(self.rollouts):
        f.write(f'\n-----------Rollout {i}----------------\n')
        f.write(str(r))

        

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
  def __str__(self):
    s = ""
    for i in range(self.pos):
      s += f"----\nstate: {self.states[i]}\naction: {self.actions[i]}\nreward: {self.rewards[i]}\ns_prime: {self.s_primes[i]}\nreturn: {self.returns[i]}\n----\n"

    return s