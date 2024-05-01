from a2c.model import q_network
import numpy as np
import torch

a1_space = np.linspace(-1, 1, 16)
a2_space = np.linspace(-1, 1, 16)

def infer(actor, observation):
    # discretization
    # eight possible actions

    action_space = np.array([[i, j] for i in a1_space for j in a2_space])
    
    with torch.no_grad():
        # print('observation', observation)

        # Convert observation to tensor
        obs = torch.from_numpy(np.array(observation)).float()

        # Create batch of actions
        actions = torch.from_numpy(action_space).float()

        # Concatenate observation and actions along the batch dimension
        s_and_a = torch.cat((obs.repeat(len(action_space), 1), actions), dim=1)

        # Pass the batch of s_and_a through the actor network
        values = actor(s_and_a)

        # Find the index of the action with the highest value
        best_index = torch.argmax(values)

        # Select the best action from the action space
        best_action = action_space[best_index]

    
    return best_index, best_action.reshape(1, 2)

def epsilon_greedy_infer(q_network, observation, epsilon=0.1):
    # print('this is epsilon', epsilon)
    if np.random.rand() < epsilon:
        action_space = np.array([[i, j] for i in a1_space for j in a2_space])
        action_index = np.random.randint(0, len(action_space))
        action = action_space[action_index].reshape(1, 2)
        
        print('taking a random action', action)
        # print('taking a random action', action)
    else:
        action_index, action = infer(q_network, observation)
        print('inferring an action', action)
        
    return action_index, action
    