from a2c.model import pi_network
import numpy as np
import torch


a1_space = np.linspace(-1, 1, 16)
a2_space = np.linspace(-1, 1, 16)

# a1_space = [-0.6, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]

a2_space = a1_space.copy()

NUM_ACTIONS = len(a1_space) * len(a2_space)
# create a reverse mapping from action to index
action_to_index = {}
for i, a1 in enumerate(a1_space):
    for j, a2 in enumerate(a2_space):
        action_to_index[(a1, a2)] = i * 16 + j

def get_index_from_action(action):
    a1, a2 = action
    index = action_to_index[(a1, a2)]
    return index

def get_action_from_index(index):
    a1_index = index // 16
    a2_index = index % 16
    a1 = a1_space[a1_index]
    a2 = a2_space[a2_index]
    return np.array([a1, a2])

def infer(pi_network, observation, epsilon=0, mode='sample'):
    # discretization
    # eight possible actions
    
    if np.random.rand() < epsilon:
        a1_index = np.random.randint(0, 16)
        a2_index = np.random.randint(0, 16)
        index = a1_index * 16 + a2_index
        action = get_action_from_index(index)
        return index, action.reshape(1, 2)
    else:
        with torch.no_grad():
            # Convert observation to tensor
            # obs = torch.from_numpy(np.array(observation)).float()
            obs = torch.tensor(observation, dtype=torch.float32)

            action_logits = pi_network(obs)
            print('logits',action_logits)
            action_disbn = torch.softmax(action_logits, dim=0)
            print(action_disbn)
            action_disbn = action_disbn.numpy()
            
            if mode == 'greedy':
                index = np.argmax(action_disbn)
            elif mode == "sample":
                index = np.random.choice(range(NUM_ACTIONS), p=action_disbn)


            action = get_action_from_index(index)
        
    return index, action.reshape(1, 2)

def epsilon_greedy_infer(pi_network, observation, epsilon=0.1):
    # print('this is epsilon', epsilon)
    if np.random.rand() < epsilon:
        action_space = np.array([[i, j] for i in a1_space for j in a2_space])
        action_index = np.random.randint(0, len(action_space))
        action = action_space[action_index].reshape(1, 2)
        
        print('taking a random action', action)
        # print('taking a random action', action)
    else:
        action_index, action = infer(pi_network, observation)
        print('inferring an action', action)
        
    return action_index, action
    