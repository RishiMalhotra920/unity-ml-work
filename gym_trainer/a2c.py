
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
    self.sigma = nn.Parameter(torch.ones(num_means) * 0.5)  # Initialize sigma
    self.softplus = nn.Softplus()
    
  
  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    out = self.relu(out)

    mu_out = self.mu_head(out)
    # sigma_out = self.sigma_head(out)
    # sigma_out = self.softplus(sigma_out)
    # look into bounding the means to be between -1 and 1
    sigma_out = self.softplus(self.sigma)
    print(sigma_out.shape, mu_out.shape)
    sigma_out = sigma_out.expand_as(mu_out)
    print('sout', sigma_out)
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
  def __init__(self, env, tb_writer, n_steps):
    self.env = env
    print("action and observation shapes", env.action_space.shape, env)
    self.policy_network = PolicyNetwork(8, 64, 64, 2)
    self.value_network = NeuralNet(8, 64, 64, 1)
    self.tb_writer = tb_writer
    self.rollout_buffer = RolloutBuffer(buffer_size=n_steps, state_dim=8, action_dim=2)
    self.ep_len = 0
    self.ep_rew = 0
    self.n_steps = n_steps
    
  
  def collect_rollouts(self, s0, start_step, total_timesteps, log_interval):
    """
    """

    s_i = s0
    end = False

    step = start_step
    a0s = []
    a1s = []
    reward = 0
    log_interval_hit = False
    while not end and (step-start_step) < self.n_steps and step < total_timesteps:
      (mu_a1, mu_a2), (sigma_a1, sigma_a2) = self.policy_network(torch.tensor(s_i))

      # print('this is mu_a1, mu_a2', mu_a1, mu_a2, sigma_a1, sigma_a2)
      # TODO: clamp here to see the effects .clamp(-1, 1) at the end
      a1 = torch.distributions.Normal(mu_a1, sigma_a1).sample().numpy().clip(-1, 1)
      a2 = torch.distributions.Normal(mu_a2, sigma_a2).sample().numpy().clip(-1, 1)
      a_i = np.array([a1, a2])
      # print('this is a_i', a_i)
      
      s_i_prime, r_i, end, _ = self.env.step(a_i)

      # print('this is s_i_prime', a_i, s_i_prime, r_i, end)

      self.rollout_buffer.add(s_i, a_i, r_i, s_i_prime)
      s_i = s_i_prime
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

  def learn(self, gamma=0.99, total_timesteps=1000, ent_coef=10e-4, policy_network_lr=7e-4, value_network_lr=7e-4, log_interval=100):

    print('params', list(self.policy_network.parameters()))
    policy_network_optim = torch.optim.RMSprop(self.policy_network.parameters(), lr=policy_network_lr)
    value_network_optim = torch.optim.RMSprop(self.value_network.parameters(), lr=value_network_lr)
    step = 0
    s0 = self.env.reset()

    while step < total_timesteps:
      
      self.rollout_buffer.reset()
      step, next_rollout_start_state, log_interval_hit = self.collect_rollouts(s0, step, total_timesteps, log_interval)
      s0 = next_rollout_start_state
      
      self.rollout_buffer.compute_returns(self.value_network, gamma)
      
      states, actions, rewards, s_primes, returns, end = self.rollout_buffer.get_rollout_tensors()
      print('this is states', states)
      mus, sigmas = self.policy_network(states)
      mu_a1, mu_a2 = mus[:, 0], mus[:, 1]
      sigma_a1, sigma_a2 = sigmas[:, 0], sigmas[:, 1]

      print('this is mu_a1, mu_a2', mu_a1, mu_a2, sigma_a1, sigma_a2)
      v = self.value_network(states).squeeze()
      print('this is v', v)

      log_probs = torch.distributions.Normal(mu_a1, sigma_a1).log_prob(actions[:, 0]) + \
                  torch.distributions.Normal(mu_a2, sigma_a2).log_prob(actions[:, 1])

      advantages = returns - v.detach()
      entropy_loss_a1 = -0.5 * (torch.log(2*torch.pi*(sigma_a1)**2) + 1).mean() #negated the entropy loss
      entropy_loss_a2 = -0.5 * (torch.log(2*torch.pi*(sigma_a2)**2) + 1).mean() #negated the entropy loss
      log_prob_advantages = -(log_probs * advantages).mean()
      policy_loss = log_prob_advantages + ent_coef * (entropy_loss_a1 + entropy_loss_a2) #reduce the loss for high entropy.
      value_loss = F.mse_loss(returns, v)
      # visualize value as a function of the ball position.

      policy_network_optim.zero_grad()
      policy_loss.backward()
      policy_network_optim.step()

      value_network_optim.zero_grad()
      value_loss.backward()
      value_network_optim.step()


      print(self.policy_network.sigma.grad, 'grads')


      if log_interval_hit:
        self.tb_writer.add_scalar('train/policy_loss', policy_loss, step)
        self.tb_writer.add_scalar('train/value_loss', value_loss, step)
        self.tb_writer.add_scalar('train/sigma_a1s', sigma_a1.mean(), step)
        self.tb_writer.add_scalar('train/sigma_a2s', sigma_a2.mean(), step)
        self.tb_writer.add_scalar('train/entropy_loss', entropy_loss_a1 + entropy_loss_a2, step)
        


      
        

        
    
    



    
