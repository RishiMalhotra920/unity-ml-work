import torch
from torch import nn
import random
def train(data, q_network, num_epochs, writer, step, save_path):
    
    # Define the loss function
    criterion = nn.MSELoss()
    # Define the optimizer
    optimizer = torch.optim.Adam(q_network.parameters(), lr=0.01)

    print("Training...")
    # there should be a dataset, dataloader and what not here
    # Train the network
    
    # shuffle the order of data
    random.shuffle(data)

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (s, a, r, s_prime) in enumerate(data):
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).reshape(2,).float()

            # Forward pass
            s_and_a = torch.cat((s, a), 0)
            r_pred = q_network(s_and_a)
            loss = criterion(r_pred, torch.tensor([r], dtype=torch.float32))

            # optimize the actor
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(data)
        
        step_epoch = step*epoch + epoch
        writer.add_scalar('training loss', avg_loss, step_epoch)

        

    print('Finished Training')
    print(f"Writing model to {save_path}")
    torch.save(q_network.state_dict(), save_path)

    return q_network
