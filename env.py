import numpy as np
from gymnasium import spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

class Environment:
  """
  Documentation: https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Python-LLAPI.md
  Consider creating your own environment without unity ml agents by following this video:
  https://www.youtube.com/watch?v=FqNpVLKSFJg. when building custom environments, this is the way to go.
  email the author and learn about possible disadvantages of this approach - maybe there's some missing optimizations?
  """

  def __init__(self, seed, clamp_continuous_action_space=True, reset_action=np.array([[-100, -100]])):
    """
    if clamp_continuous_action_space is True, we restrict actions between -1 and 1. by default unity takes spaces between -inf and +inf
    However, in environment files like BallAgent3D.cs, you can see that the action space is restricted between -1 and 1
    """
    self.unity_env = UnityEnvironment(seed=seed)
    self.unity_env.reset() #you need to reset to access the behaviors below
    self.reset_action = reset_action
    
    # Get the behavior names from the environment
    behavior_names = list(self.unity_env.behavior_specs.keys())
    if len(behavior_names) > 1:
        raise ValueError("This environment has multiple behavior names. This class only supports environments with a single behavior name.")
    elif len(behavior_names) == 0:
        raise ValueError("No behavior names found in the environment.")
    else:
        self.behavior_name = behavior_names[0]
    
    # Get the behavior spec for the selected behavior name
    behavior_spec = self.unity_env.behavior_specs[self.behavior_name]
    
    low = -1 if clamp_continuous_action_space else -np.inf
    high = 1 if clamp_continuous_action_space else np.inf

    # Set the action space
    if behavior_spec.action_spec.is_continuous():
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(behavior_spec.action_spec.continuous_size,), 
            dtype=np.float32
        )
    else:
        self.action_space = spaces.MultiDiscrete(behavior_spec.action_spec.discrete_branches)
    
    # Set the observation space
    self.observation_space = spaces.Box(
        low=-np.inf, 
        high=np.inf, 
        shape=behavior_spec.observation_specs[0].shape, 
        dtype=np.float32
    )
    
    
    decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
    
    self.num_agents = len(decision_steps) + len(terminal_steps)
    
    
    
    
    
    assert self.num_agents == 12

  def reset(self):
    """
    reset the full environment
    """
    self.unity_env.reset()

  def reset_agent(self, agent_id):
    """
    reset the agent
    """
    self.unity_env.set_action_for_agent(self.behavior_name, 
                                        agent_id,
                                        ActionTuple(continuous=self.reset_action))
    
  
  def step_agent(self, agent_id, action):
    """
    step one agent

    Args:
      action: the action to take
    
    Returns:
      observation: the observation after the action for that agent
      reward: the reward after the action
      terminated: whether the episode is terminated
      info: additional information
    """
    
    self.unity_env.set_action_for_agent(self.behavior_name, agent_id, ActionTuple(action))
    
    decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

    decision_step = decision_steps.get(agent_id, None)
    
    if decision_step:
      observation = decision_step.obs[0]
      reward = decision_step.reward
      terminated = np.float32(1)
    else:
      terminal_step = terminal_steps.get(agent_id, None)
      observation = terminal_step.obs[0]
      reward = terminal_step.reward
      terminated = np.float32(0)


    return observation, reward, terminated, {}
    


  def step(self, actions):
    """
    takes in an action for all agents and returns the observations, rewards, terminated and infos for all agents

    Args:
      actions: a batch of actions of shape (num_agents, action_dim)
    
    Returns:
      observations (np.array(num_agents, obs_dim)): the observations after the action for all agents
      rewards (np.array(num_agents, 1)): the rewards after the action for all agents
      terminated (np.array(num_agents, 1)): whether the episode is terminated for all agents
      infos (dict): additional information for all agents. contains the agent_id as the key and the info as the value
    """
    self.unity_env.set_actions(self.behavior_name, ActionTuple(actions))
    decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

    observations = np.zeros((self.num_agents, *self.observation_space.shape))
    
    rewards = np.zeros((self.num_agents, 1))
    terminated = np.zeros((self.num_agents, 1))
    infos = {}
    for agent_id, decision_step in decision_steps.items():
      
      
      
      observations[agent_id] = decision_step.obs[0]

      rewards[agent_id] = decision_step.reward
      terminated[agent_id] = 0
      infos[agent_id] = {}
    
    for agent_id, terminal_step in terminal_steps.items():

      observations[agent_id] = terminal_step.obs[0]
      rewards[agent_id] = terminal_step.reward
      terminated[agent_id] = 1
      infos[agent_id] = {}
    
    return observations, rewards, terminated, infos

  def close(self):
    """
    close the environment
    """
    self.unity_env.close()
    

if __name__ == "__main__":
  myenv = Environment(seed=42)
  # test all methods above and log stuff

  # test reset
  print("testing reset")
  myenv.reset()

  # test reset_agent
  print("testing reset_agent")
  myenv.reset_agent(2)

  # test step_agent
  print("testing step_agent")
  action = np.array([[0.5, 0.5]])
  observation, reward, terminated, info = myenv.step_agent(0, action)
  print(observation, reward, terminated, info)

  # test step with 12 agents
  print("testing step")
  actions = np.random.uniform(low=-1, high=1, size=(12, 2))
  print('actions', actions)
  observations, rewards, terminated, infos = myenv.step(actions)
  print(observations, rewards, terminated, infos)

  # test close
  print("testing close")
  myenv.close()
