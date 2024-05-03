import torch
from torch import nn
import random
import numpy  as np 
from a2c.inference import get_index_from_action

def train(data, pi_network, gamma, lr, num_epochs, writer, step, save_path):
    
    # Define the loss function
    criterion = nn.MSELoss()
    # Define the optimizer
    optimizer = torch.optim.Adam(pi_network.parameters(), lr=lr)

    print("Training...")
    # there should be a dataset, dataloader and what not here
    # Train the network
    
    # shuffle the order of data
    random.shuffle(data) #shuffle whole trajectories, maintaining the order of each trajectory

    for epoch in range(num_epochs):
        total_loss = 0
        num_updates = 0
        total_G = 0
        for episode in data:
            G = 0
            for t in range(len(episode)-1, -1, -1):
                (s, a, r, s_prime) = episode[t]
                s = torch.from_numpy(s).float()

                G = r + gamma*G #this computation is duplicated per epoch, but it's fine for now
                # Forward pass
                a_logits = pi_network(s)

                # print('a_logits', a_logits, a_logits.shape)
                # input()

                # log softmax more stable due to the log sum exp trick.
                a_disbn = torch.log_softmax(a_logits, dim=0) 

                # print('this is a', a)
                # a is a vector of one action.
                index = get_index_from_action(a[0])
                pi_a_given_s = a_disbn[index]
                # print('this is the pi_a_given_s', index, pi_a_given_s)

                print('this is a_disbn', a_disbn)

                steps_to_go = len(episode) - 1 - t #good save by gpt
                loss = -(gamma**steps_to_go) * G * pi_a_given_s #add a negative sign to minimize the negative...

                print('this is the loss', loss, gamma, G, pi_a_given_s)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_updates += 1
            
            total_G += G
            
        
        avg_loss = total_loss / num_updates
        
        step_epoch = step*num_epochs + epoch
        writer.add_scalar('episode return', total_G/len(data), step_epoch)
        writer.add_scalar('training loss', avg_loss, step_epoch)

        

    print('Finished Training')
    print(f"Writing model to {save_path}")
    torch.save(pi_network.state_dict(), save_path)

    return pi_network
