from a2c.model import actor, critic
import numpy as np
import torch

def infer(actor, observation):
    with torch.no_grad():
        # print('observation', observation)
        obs = torch.from_numpy(np.array(observation))
        preds = actor(obs)

        return preds.detach().numpy().reshape(1, 2)

def epsilon_greedy_infer(actor, observation, epsilon=0.1):
    # print('this is epsilon', epsilon)
    if np.random.rand() < epsilon:
        action = np.random.uniform(-1, 1, (1, 2))
        # print('taking a random action', action)
    else:
        action = infer(actor, observation)
        
    return action
    