
# from neural_net import NeuralNet
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from gym_trainer.rolloutbuffer import Rollouts
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
  # Note! you have to specify the lr for any new params you add.
  def __init__(self, input_size, hidden_size, hidden_size2, num_means):
    super(PolicyNetwork, self).__init__()

    self.policy_network = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
    )
    self.mu_head = nn.Sequential(
      nn.Linear(hidden_size, num_means),
      # try adding the tanh non linearity here. may be game changin?
      # nn.Tanh()
    )
    self.variance_head = nn.Sequential(
      nn.Linear(hidden_size, 1),
      nn.Softplus(),
    )
  
  def forward(self, x):

    embedding = self.policy_network(x)
    mu_out = self.mu_head(embedding)
    variance_out = self.variance_head(embedding)
    sigma_out = torch.sqrt(variance_out)

    return mu_out, sigma_out

# TODO: optimizer: self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

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
  def __init__(self, env, tb_writer, n_steps, num_rollouts_per_update, load_policy_network_checkpoint_path=None, load_value_network_checkpoint_path=None):
    self.env = env
    # print("action and observation shapes", env.action_space.shape, env)
    self.policy_network = PolicyNetwork(8, 64, 64, 2)
    self.value_network = NeuralNet(8, 64, 64, 1)
    # load policy network weihgts
    if load_policy_network_checkpoint_path:
      self.policy_network.load_state_dict(torch.load(load_policy_network_checkpoint_path))
    
    if load_value_network_checkpoint_path:
      self.value_network.load_state_dict(torch.load(load_value_network_checkpoint_path))

    
    self.tb_writer = tb_writer
    self.rollout_buffers = Rollouts(num_rollouts_per_update, n_steps, 8, 2)
    self.ep_len = 0
    self.ep_rew = 0
    self.n_steps = n_steps
    self.num_rollouts_per_update = num_rollouts_per_update
    
  
  def collect_rollouts(self, s0, start_step, total_timesteps, log_interval):
    """
    """

    log_interval_hit = False
    total_steps_taken = 0
    s_i = s0
    for rollout_idx in range(self.num_rollouts_per_update):
      end = False
      step = start_step
      while not end and (step-start_step) < self.n_steps and step < total_timesteps:
        (mu_a1, mu_a2), sigma = self.policy_network(torch.tensor(s_i))
        # sigma = sigma.reshape(-1)

        print('trace 2: this is mu', mu_a1, mu_a2, sigma)
        print('trace 3: this is torch out', self.policy_network(torch.tensor(s_i)))

        # print('this is mu_a1, mu_a2', mu_a1, mu_a2, sigma_a1, sigma_a2)
        # TODO: clamp here to see the effects .clamp(-1, 1) at the end
        a1 = torch.distributions.Normal(mu_a1, sigma).sample().numpy().clip(-1, 1)
        a2 = torch.distributions.Normal(mu_a2, sigma).sample().numpy().clip(-1, 1)
        a_i = np.array([a1, a2]).reshape(-1)
        print('this is a_i', a_i)
        
        s_i_prime, r_i, end, _ = self.env.step(a_i)

        # print('this is s_i_prime', a_i, s_i_prime, r_i, end)

        self.rollout_buffers.add(rollout_idx, s_i, a_i, r_i, s_i_prime)

        s_i = s_i_prime
        
        step += 1
        total_steps_taken += 1
        self.ep_len += 1
        self.ep_rew += r_i

        if step % log_interval == 0:
          log_interval_hit = True
      
      if end:
        s_i = self.env.reset()
        self.tb_writer.add_scalar('rollout/ep_rew_mean', self.ep_rew/self.ep_len, step)
        self.tb_writer.add_scalar('rollout/ep_len_mean', self.ep_len, step)
        self.ep_rew = 0
        self.ep_len = 0
      else:
        s_i = s_i_prime
      

      self.rollout_buffers.set_end(rollout_idx, end)
    
    final_step = start_step + total_steps_taken

    # if log_interval_hit:
      # Create a scatter plot
      # fig, ax = plt.subplots()
      # ax.scatter(a0s, a1s)
      # ax.set_title('Action Scatter Plot')
      # ax.set_xlabel('Action 0')
      # ax.set_ylabel('Action 1')

      # create this for 100s of actions
      # Add the figure to TensorBoard
      # self.tb_writer.add_figure('rollout/actions', fig, global_step=step)

    
    return final_step, s_i, log_interval_hit
  
  def collect_rollouts_for_inference(self, num_episodes, deterministic):
    """
    """
    episode_lengths = []
    for _ in range(num_episodes):
      s_i = self.env.reset()
      end = False
      step = 0
      while not end:
        (mu_a1, mu_a2), sigma = self.policy_network(torch.tensor(s_i))
        if deterministic:
          a1 = mu_a1.detach().numpy().clip(-1, 1)
          a2 = mu_a2.detach().numpy().clip(-1, 1)
        else:
          a1 = torch.distributions.Normal(mu_a1, sigma).sample().numpy().clip(-1, 1)
          a2 = torch.distributions.Normal(mu_a2, sigma).sample().numpy().clip(-1, 1)
        a_i = np.array([a1, a2])
        s_i_prime, r_i, end, _ = self.env.step(a_i)
        s_i = s_i_prime
        step += 1
      episode_lengths.append(step)
    return np.mean(episode_lengths), np.std(episode_lengths)

  
  # the rollout approach needs to be more sophisticated.
# you rollout 5 steps with pi, then update pi and v with the rollout
# then rollout 5 more steps from the last step with pi, then update pi and v with the rollout
# repeat until you reach the end of the trajectory, then keep going until you hit num_steps.
# tomorrow it will take 2-3 hours to code this out but you can get it done.

  def get_sigma(self, step):
    """
    """
    schedule = {
      0: 1.0,
      5000: 0.7,
      10000: 0.3,
      15000: 0.1,
      20000: 0.01
    }
    # schedule = {
    #   0: 0.1
    # }
    for k, v in schedule.items():
      if step >= k:
        sigma = v
    
    return torch.tensor(sigma)

  def learn(self, *, checkpoint_path, gamma=0.99, total_timesteps=1000, ent_coef=10e-4, policy_network_lr=7e-4, value_network_lr=7e-4, sigma_lr=1e-4,log_interval=100, max_grad_norm=0.5):

    # print('params', list(self.policy_network.parameters()))
    
    # could have also added value params here into one optimizer.
    # then wouldn't have to clear twice.
    # policy_network_optim = torch.optim.RMSprop([
    #   {'params': self.policy_network.sigma_network.parameters(), 'lr': policy_network_lr},
    #   {'params': self.policy_network.mu_network.parameters(), 'lr': policy_network_lr}
    # ])
    policy_network_optim = torch.optim.RMSprop(self.policy_network.parameters(), lr=policy_network_lr)
    value_network_optim = torch.optim.RMSprop(self.value_network.parameters(), lr=value_network_lr)
    step = 0
    s0 = self.env.reset()

    while step < total_timesteps:
      self.rollout_buffers.write()
      self.rollout_buffers.reset()

      step, next_rollout_start_state, log_interval_hit = self.collect_rollouts(s0, step, total_timesteps, log_interval)
      s0 = next_rollout_start_state

      self.rollout_buffers.compute_returns(self.value_network, gamma)
      
      states_batch, actions_batch, rewards_batch, s_primes_batch, returns_batch, values_batch, advantages_batch, end = self.rollout_buffers.get_rollout_tensors()

      print('shapes:', states_batch.shape, actions_batch.shape, rewards_batch.shape, s_primes_batch.shape, returns_batch.shape, end)
      # print('this is states', states)
      mus, sigma = self.policy_network(states_batch)
      mu_a1, mu_a2 = mus[:, 0], mus[:, 1]
      sigma = sigma.reshape(-1)

      # print('this is mu_a1, mu_a2', mu_a1, mu_a2, sigma_a1, sigma_a2)
      v = self.value_network(states_batch).reshape(-1)
      # print('this is v', v)

      # sigma_a1 = sigma_a2 = self.get_sigma(step)

      log_probs = torch.distributions.Normal(mu_a1, sigma).log_prob(actions_batch[:, 0]) + \
                  torch.distributions.Normal(mu_a2, sigma).log_prob(actions_batch[:, 1])

      advantages = returns_batch - v.detach()
      entropy_loss = ent_coef * -0.5 * (torch.log(2*torch.pi*(sigma)**2) + 1).mean() #negated the entropy loss
      # entropy_loss_a2 = -0.5 * (torch.log(2*torch.pi*(sigma_a2)**2) + 1).mean() #negated the entropy loss
      log_prob_advantages = -(log_probs * advantages).mean()
      policy_loss = log_prob_advantages + (entropy_loss) #reduce the loss for high entropy.
      # print('this is value_loss', returns, v)
      value_loss = F.mse_loss(returns_batch, v)
      # visualize value as a function of the ball position.

      policy_network_optim.zero_grad()
      policy_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_grad_norm)
      # print('sigma: ', sigma_a1, sigma_a2)
      policy_network_optim.step()

      value_network_optim.zero_grad()
      value_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_grad_norm)
      value_network_optim.step()


      if log_interval_hit:
        self.tb_writer.add_scalar('train/policy_loss', policy_loss, step)
        self.tb_writer.add_scalar('train/value_loss', value_loss, step)
        self.tb_writer.add_scalar('train/sigma_a1s', sigma.mean(), step)
        # self.tb_writer.add_scalar('train/sigma_a2s', sigma_a2.mean(), step)
        self.tb_writer.add_scalar('train/entropy_loss', entropy_loss, step)
        
        (checkpoint_path / f"step_{step}").mkdir(parents=True, exist_ok=True)

        policy_network_path = checkpoint_path / f"step_{step}" / "policy_network.pth"
        value_network_path = checkpoint_path / f"step_{step}" / "value_network.pth"

        print('saving policy_network to ', policy_network_path)
        print('saving value_network to ', value_network_path)
        
        torch.save(self.policy_network.state_dict(), checkpoint_path / f"step_{step}" / "policy_network.pth")
        torch.save(self.value_network.state_dict(),  checkpoint_path / f"step_{step}" / "value_network.pth")
        


      
        

        
    
    



    
