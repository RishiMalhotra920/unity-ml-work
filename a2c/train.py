import torch
from torch import nn
import random
import numpy  as np 

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
                logits = pi_network(s)
                print('this is s', s)
                print('this is logits', logits)
                a1_mean, a2_mean, a1_var, a2_var = logits
                softplus = torch.nn.Softplus()
                a1_var = softplus(a1_var)
                a2_var = softplus(a2_var)


                actual_a1, actual_a2 = torch.tensor(a[0][0]), torch.tensor(a[0][1])

                pi_a_given_s = torch.distributions.Normal(a1_mean, a1_var).log_prob(actual_a1) + torch.distributions.Normal(a2_mean, a2_var).log_prob(actual_a2)


                # a is a vector of one action.
                # index = get_index_from_action(a[0])
                # pi_a_given_s = a_disbn[index]
                # print("index", index, len(a_disbn))
                # print('this is the pi_a_given_s', index, pi_a_given_s)

                # print('this is a_disbn', a_disbn)

                steps_to_go = len(episode) - 1 - t #good save by gpt
                loss = -(gamma**steps_to_go) * G * pi_a_given_s #add a negative sign to minimize the negative...

                # print('this is the loss', loss, gamma, G, pi_a_given_s)

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
