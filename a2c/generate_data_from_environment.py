import numpy as np
from mlagents_envs.base_env import ActionTuple


def generate_data_from_environment(policy, env, num_agents, num_episodes_per_agent, max_episode_length, writer, step, save_path):
  """
  num_episodes (int): The number of episodes to collect data for.
  in the case of many agents, the number of episodes is the total number of episodes across all agents.
  if there are 10 agents, but only 5 episodes specified, then there will be 10 episodes.
  """
  
  behavior_name = "3DBall?team=0"
  episode_memory = {}
  episode_idx = 0
  replay_buffer = [] #stores a list of trajectories
  transition_tuples = {}
  env.reset()
  print("Collecting data")
  action_indexes = []
  agent_episode_counts = {i: 0 for i in range(num_agents)}

  while True:
      decision_steps, terminal_steps = env.get_steps(behavior_name)

      # Generate the action array based on the number of active agents
      # print('transition_tuples', transition_tuples)
      for agent_id, decision_step in decision_steps.items():
          if agent_episode_counts.get(agent_id, 0) >= num_episodes_per_agent:
            # print('trace 1')
            continue
          if len(episode_memory.get(agent_id, [])) > max_episode_length:
            # print('episode memory reached max length for agent_id: ', agent_id)
            replay_buffer.append(episode_memory[agent_id])
            del episode_memory[agent_id]
            del transition_tuples[agent_id]
            reset_action = np.array([[-100, -100]])
            env.set_action_for_agent(behavior_name, agent_id, ActionTuple(continuous=reset_action))
            episode_idx += 1
            agent_episode_counts[agent_id] = agent_episode_counts.get(agent_id, 0) + 1
            continue

          observations = decision_step.obs
          observation = observations[0]
          reward = decision_step.reward

          # print("this is", agent_id, transition_tuples.get(agent_id, None))
          if agent_id in transition_tuples:
            transition_tuples[agent_id][2] = reward
            transition_tuples[agent_id][3] = observation
            episode_memory[agent_id] = episode_memory.get(agent_id, []) + [transition_tuples[agent_id]]
            # print('added to episode memory', episode_memory)

          # print(f"Agent {agent_id} has received a decision step reward of {reward}.")
          # TODO: we can batch the actions across many agents for efficiency.
          action_index, actions = policy(observation)
          action_indexes.append(action_index)
          
          # actions = np.random.uniform(low=-1, high=1, size=(num_agents, 2))
          action_tuple = ActionTuple(continuous=actions)
          # print('these are actions', actions)
          transition_tuples[agent_id] = [observation, actions, None, None]
          env.set_action_for_agent(behavior_name, agent_id, action_tuple) # should be set_actions_for_agent here...


      # when there is a terminal step for an agent, switch out rewards 
      # for that agent with returns and then clear the agent's episode memory and reset that agent.
      for agent_id, terminal_step in terminal_steps.items():
          if agent_episode_counts.get(agent_id, 0) >= num_episodes_per_agent:
            # print('trace 2')
            continue
          if agent_id in transition_tuples:
            # print('in terminal step', agent_id, transition_tuples)
            observation = terminal_step.obs[0]
            reward = terminal_step.reward
            transition_tuples[agent_id][2] = reward
            transition_tuples[agent_id][3] = observation
            episode_memory[agent_id] = episode_memory.get(agent_id, []) + [transition_tuples[agent_id]]
            
            replay_buffer.append(episode_memory[agent_id])
            
            del episode_memory[agent_id]
            del transition_tuples[agent_id]

            episode_idx += 1
            agent_episode_counts[agent_id] = agent_episode_counts.get(agent_id, 0) + 1
      
      
      # print('agent_episode_counts', agent_episode_counts, len(agent_episode_counts))
      if len(agent_episode_counts) > 0:
        are_all_agents_done = True
        for agent_id in agent_episode_counts:
          print('agent_id', agent_id, agent_episode_counts.get(agent_id, 0))
          if agent_episode_counts.get(agent_id, 0) < num_episodes_per_agent:
            # print('in here', agent_id, agent_episode_counts)
            are_all_agents_done = False
            break
        if are_all_agents_done:
          break
    
      env.step() 

  print("Data collected")
  writer.add_histogram('data collection action values', np.array(action_indexes), step, bins=np.arange(-0.5, 32*32+5, 1))
  writer.add_scalar("average episode length", np.mean([len(episode) for episode in replay_buffer]), step)

  if save_path != None:
    with open(save_path, 'w') as f:
      f.write(str(replay_buffer))
  
  return replay_buffer

  