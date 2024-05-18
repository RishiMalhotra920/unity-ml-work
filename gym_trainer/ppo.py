
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
    self.std_param = nn.Parameter(torch.tensor([1.]))
    self.softplus = nn.Softplus()
  
  def forward(self, x):

    embedding = self.policy_network(x)
    mu_out = self.mu_head(embedding)

    # print('mu_out', mu_out.shape)

    std_out = self.softplus(self.std_param)
    variance_out = torch.square(std_out)
    expanded_variance_out = variance_out.expand(mu_out.shape[0], mu_out.shape[1])
    # print('this is expanded_variance_out', variance_out, expanded_variance_out)
    cov_mat = torch.diag_embed(expanded_variance_out)

    # print('the mu and cov', mu_out, cov_mat)
    distribution = torch.distributions.MultivariateNormal(mu_out, cov_mat)

    return distribution

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


class PPO():
  def __init__(self, env, tb_writer, n_steps, load_policy_network_checkpoint_path=None, load_value_network_checkpoint_path=None):
    self.env = env
    self.load_policy_network_checkpoint_path = load_policy_network_checkpoint_path
    self.policy_network = self.init_policy_network()
    self.policy_network_old = self.init_policy_network()
    
    self.value_network = NeuralNet(8, 64, 64, 1)

    if load_value_network_checkpoint_path:
      value_network.load_state_dict(torch.load(load_value_network_checkpoint_path))
    
    self.tb_writer = tb_writer
    self.rollout_buffers = Rollouts(env.num_agents, n_steps, 8, 2)
    
    # this may have to be in its own Agent class at some point
    # access by agent_memory[agent_idx]["ep_rew"]
    self.agent_memory = [{"ep_rew": 0, "ep_len": 0} for _ in range(env.num_agents)]
    self.n_steps = n_steps


  def init_policy_network(self):
    policy_network = PolicyNetwork(8, 64, 64, 2)
    
    if self.load_policy_network_checkpoint_path:
      policy_network.load_state_dict(torch.load(self.load_policy_network_checkpoint_path))
    
    return policy_network
  
  
  def collect_rollouts(self, start_states, start_step, total_timesteps, gamma, log_interval):
    """
    Collect rollouts for all agents in the environment. Each agent is stepped
    for a max of n_steps or until the environment terminates, 
    whichever comes first. Stores rollouts in self.rollout_buffers

    Args:
      start_states: np.array of shape (num_agents, state_dim): the starting states for all agents
      start_step: int: the starting step
      total_timesteps: int: the total number of timesteps to run for
      log_interval: int: the interval at which to log data to TensorBoard. TODO: probs move this to the __init__.
    
    Returns:
      final_step: int: the final step after all agents have been stepped
      end_states: np.array of shape (num_agents, state_dim): the final states after all agents have been stepped
      log_interval_hit: bool: whether the log_interval has been hit
    """
    print(f"Collecting rollouts {start_step}/{total_timesteps} [{round(start_step/total_timesteps, 2) * 100}%]")
    self.rollout_buffers.reset()

    # end_states = np.zeros((self.env.num_agents, self.env.observation_space.shape[0]))
    log_interval_hit = False
    total_steps_taken = 0
    
    for agent in range(self.env.num_agents):
      end = False
      step = start_step
      s_i = start_states[agent]
      # print('this is s_i', s_i)
      # run each for max of t timesteps
      while not end and (step-start_step) < self.n_steps and step < total_timesteps:
        print('step', step)
        state_batch = torch.tensor(s_i, dtype=torch.float32).reshape(1, -1)
        action_distribution = self.policy_network(state_batch)
        # print('a disbn', action_distribution)
        # print('this is action_distribution', action_distribution.sample())
        a_i = action_distribution.sample().numpy().clip(-1, 1).reshape(-1)
        s_i_prime, r_i, end, _ = self.env.step_agent(agent, a_i)
        print('this is end:', end)

        self.rollout_buffers.add(agent, s_i, a_i, r_i, s_i_prime)
        s_i = s_i_prime
        step += 1
        total_steps_taken += 1
        self.agent_memory[agent]["ep_rew"] += r_i
        self.agent_memory[agent]["ep_len"] += 1

        if step % log_interval == 0:
          log_interval_hit = True

      # if end:
        # end_states[agent], _ = self.env.reset_agent(agent)
      # else:
        # end_states[agent], _ = s_i_prime
      
      self.rollout_buffers.set_end(agent, end)
    
    end_states = self.env.reset()
    print('the end states', end_states)

    total_reward = 0
    total_ep_len = 0
    for agent in range(self.env.num_agents):
      total_reward += self.agent_memory[agent]["ep_rew"]
      total_ep_len += self.agent_memory[agent]["ep_len"]
    
    self.tb_writer.add_scalar('rollout/ep_rew_mean', total_reward, step)
    self.tb_writer.add_scalar('rollout/ep_len_mean', total_ep_len, step)
    
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

    self.rollout_buffers.compute_returns(self.value_network, gamma)

    return final_step, end_states, log_interval_hit


  def learn(self, *, checkpoint_path, gamma=0.99, total_timesteps=1000, value_coef=0.5, ent_coef=10e-4, policy_network_lr=7e-4, value_network_lr=7e-4, sigma_lr=1e-4,log_interval=100, max_grad_norm=0.5):


    optimizer = torch.optim.RMSprop([
        {'params': self.policy_network.parameters(), 'lr': policy_network_lr},
        {'params': self.value_network.parameters(), 'lr': value_network_lr}
    ])
    step = 0
    
    start_states, _ = self.env.reset()

    while step < total_timesteps:

      step, next_rollout_start_states, log_interval_hit = self.collect_rollouts(start_states, step, total_timesteps, gamma, log_interval)
      start_states = next_rollout_start_states
      states_batch, actions_batch, rewards_batch, s_primes_batch, returns_batch, values_batch, advantages_batch, end = self.rollout_buffers.get_rollout_tensors()

      # print('shapes:', states_batch.shape, actions_batch.shape, rewards_batch.shape, s_primes_batch.shape, returns_batch.shape, end)

      distribution = self.policy_network(states_batch)
      log_probs = distribution.log_prob(actions_batch)
      sigma = torch.sum(distribution.stddev)
      # print('distribution.stddev', distribution.stddev, sigma)

      old_distribution = self.policy_network_old(states_batch)
      old_log_probs = old_distribution.log_prob(actions_batch)

      pi_ratio = torch.exp(log_probs - old_log_probs)

      pi_ratio_clip = torch.clamp(pi_ratio, 1-0.2, 1+0.2) * advantages_batch

      l_clip = -torch.min(pi_ratio*advantages_batch, pi_ratio_clip*advantages_batch).mean()

      l_vf = F.mse_loss(values_batch, returns_batch)

      entropy_loss = ent_coef * -0.5 * (torch.log(2*torch.pi*(sigma)**2) + 1).mean() #negated the entropy loss

      # print('pi_ratio', pi_ratio, 'pi_ratio_clip', pi_ratio_clip, 'advantages_batch', advantages_batch, 'l_clip', l_clip, 'l_vf', l_vf)
      # print('entropy_loss', entropy_loss)
      loss = l_clip + value_coef * l_vf + ent_coef * entropy_loss

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_grad_norm)
      torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_grad_norm)
      optimizer.step()


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
    
    # swaps the networks
    self.policy_network_old = self.init_policy_network()
    self.policy_network_old.load_state_dict(self.policy_network.state_dict())
    self.policy_network = self.init_policy_network()        


  def collect_rollouts_for_inference(self, num_episodes, deterministic):
    """
    """
    episode_lengths = []
    for _ in range(num_episodes):
      s_i, _ = self.env.reset()
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
