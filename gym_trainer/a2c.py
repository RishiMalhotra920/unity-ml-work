
# from neural_net import NeuralNet
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from gym_trainer.rolloutbuffer import RolloutBuffer
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, hidden_size2, num_means):
    super(PolicyNetwork, self).__init__()

    self.l1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size, hidden_size2)
    self.mu_head = nn.Linear(hidden_size2, num_means)
    self.sigma_head = nn.Linear(hidden_size2, num_means)
    self.softplus = nn.Softplus()
    
  
  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    out = self.relu(out)

    mu_out = self.mu_head(out)
    sigma_out = self.sigma_head(out)
    sigma_out = self.softplus(sigma_out)
    # look into bounding the means to be between -1 and 1

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


class MyA2C():
  def __init__(self, env, tb_writer):
    self.env = env
    print("action and observation shapes", env.action_space.shape, env)
    self.policy_network = PolicyNetwork(8, 64, 64, 2)
    self.value_network = NeuralNet(8, 64, 64, 1)
    self.tb_writer = tb_writer
    self.rollout_buffer = RolloutBuffer()
    self.ep_len = 0
    self.ep_rew = 0
  
  def collect_rollouts(self, s0, start_step, n_steps, total_timesteps, log_interval):
    """
    n_steps: total number of steps to simulate
    """

    s_i = s0
    end = False

    step = start_step
    a0s = []
    a1s = []
    reward = 0
    log_interval_hit = False
    while not end and (step-start_step) < n_steps and step < total_timesteps:
      (mu_a1, mu_a2), (sigma_a1, sigma_a2) = self.policy_network(torch.tensor(s_i))

      # print('this is mu_a1, mu_a2', mu_a1, mu_a2, sigma_a1, sigma_a2)
      a1 = torch.distributions.Normal(mu_a1, sigma_a1).sample()
      a2 = torch.distributions.Normal(mu_a2, sigma_a2).sample()
      a_i = (a1, a2)
      # print('this is a_i', a_i)
      
      s_i_prime, r_i, end, _ = self.env.step(a_i)

      # print('this is s_i_prime', a_i, s_i_prime, r_i, end)

      self.rollout_buffer.add(s_i, a_i, r_i, s_i_prime)
      a0s.append(a1.item())
      a1s.append(a2.item())
      
      reward += r_i
      step += 1
      self.ep_len += 1
      self.ep_rew += r_i

      if step % log_interval == 0:
        log_interval_hit = True
    
    if end:
      next_rollout_start_state = self.env.reset()
      self.tb_writer.add_scalar('rollout/ep_rew_mean', self.ep_rew/self.ep_len, step)
      self.tb_writer.add_scalar('rollout/ep_len_mean', self.ep_len, step)
      self.ep_rew = 0
      self.ep_len = 0
    else:
      next_rollout_start_state = s_i_prime
      
    self.rollout_buffer.set_end(end)

    if log_interval_hit:
      # Create a scatter plot
      fig, ax = plt.subplots()
      ax.scatter(a0s, a1s)
      ax.set_title('Action Scatter Plot')
      ax.set_xlabel('Action 0')
      ax.set_ylabel('Action 1')

      
      # Add the figure to TensorBoard
      self.tb_writer.add_figure('rollout/actions', fig, global_step=step)

    
    return step, next_rollout_start_state, log_interval_hit
  
  # the rollout approach needs to be more sophisticated.
# you rollout 5 steps with pi, then update pi and v with the rollout
# then rollout 5 more steps from the last step with pi, then update pi and v with the rollout
# repeat until you reach the end of the trajectory, then keep going until you hit num_steps.
# tomorrow it will take 2-3 hours to code this out but you can get it done.

  def learn(self, gamma=0.99, n_steps=5, total_timesteps=1000, ent_coef=10e-4, policy_network_lr=7e-4, value_network_lr=7e-4, log_interval=100):

    policy_network_optim = torch.optim.RMSprop(self.policy_network.parameters(), lr=policy_network_lr)
    value_network_optim = torch.optim.RMSprop(self.value_network.parameters(), lr=value_network_lr)
    step = 0
    s0 = self.env.reset()

    while step < total_timesteps:
      
      self.rollout_buffer.reset()
      step, next_rollout_start_state, log_interval_hit = self.collect_rollouts(s0, step, n_steps, total_timesteps, log_interval)
      s0 = next_rollout_start_state
      
      if self.rollout_buffer.get_end():
        R = torch.tensor([0])
      else:
        s_last = self.rollout_buffer.get_last_state()
        R = self.value_network(torch.tensor(s_last))

      value_loss = 0
      policy_loss = 0
      sigma_a1s = 0
      sigma_a2s = 0
      total_entropy_bonus = 0
      for i in range(len(self.rollout_buffer)-1, -1, -1):
        # print('this is self.rollout_buffer[i]', self.rollout_buffer[i])
        s_i, (a1_i, a2_i), r_i, s_i_prime = self.rollout_buffer[i]
        
        R = R.detach()
        # treat the value function as an oracle and do not backprop through it when calculating the policy loss.
        R = r_i + gamma * R
        print('this is R', R)
        a_i = self.policy_network(torch.tensor(s_i))
        v = self.value_network(torch.tensor(s_i))

        (mu_a1, mu_a2), (sigma_a1, sigma_a2) = a_i
        print('this is mu_a1, mu_a2', mu_a1, mu_a2, sigma_a1, sigma_a2)
        log_prob = torch.distributions.Normal(mu_a1, sigma_a1).log_prob(a1_i) + torch.distributions.Normal(mu_a2, sigma_a2).log_prob(a2_i)
        advantage = R - v.detach()
        entropy_bonus = 0.5 * (torch.log(2*torch.pi*(sigma_a1+sigma_a2)**2) + 1)
        print('this is advantage', advantage)
        policy_network_loss = -(log_prob * advantage) - ent_coef * entropy_bonus #reduce the loss for high entropy.

        value_network_loss = F.mse_loss(R, v)
        print('logging 1: ', log_prob, advantage, policy_network_loss, advantage)
        print('logging 2: ', v, R, value_network_loss, advantage)
        policy_network_optim.zero_grad()
        policy_network_loss.backward()
        policy_network_optim.step()
        
        value_network_optim.zero_grad()
        value_network_loss.backward()
        value_network_optim.step()

        policy_loss += policy_network_loss.detach().item()
        value_loss += value_network_loss.detach().item()
        sigma_a1s += (sigma_a1).detach().item()
        sigma_a2s += (sigma_a2).detach().item()
        total_entropy_bonus += entropy_bonus.detach().item()
        self.tb_writer.add_scalar('train/advantage', advantage, step)
        
      if log_interval_hit:
        self.tb_writer.add_scalar('train/policy_loss', policy_loss/len(self.rollout_buffer), step)
        self.tb_writer.add_scalar('train/value_loss', value_loss/len(self.rollout_buffer), step)
        self.tb_writer.add_scalar('train/sigma_a1s', sigma_a1s/len(self.rollout_buffer), step)
        self.tb_writer.add_scalar('train/sigma_a2s', sigma_a1s/len(self.rollout_buffer), step)
        self.tb_writer.add_scalar('train/entropy_bonus', total_entropy_bonus/len(self.rollout_buffer), step)
        


      
        

        
    
    



    
