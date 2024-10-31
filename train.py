from util.util import load_data_percentage, CustomDataset, DataLoader
from model.AttentionResNet56 import AttentionResNet56
from model.AttentionResNet56 import AttentionResNet56
from util.loss import circular_mse_loss, cosine_similarity_loss, combined_loss  # Adjust import as necessary
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml


# Load 10% of the data
X, Y, Z = load_data_percentage('./data/X.npy', './data/Y.npy', './data/Z.npy', percentage=100) 
# X is of shape (1848, 32768, 2)
# Y is of shape (1848, 32768, 1)
# Z is of shape (1848, 1, 2)
dataset = CustomDataset(X, Y, Z)

def train_model(dataset, model_class, config):
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    loss_type = config['loss_type']

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_shape = (X.shape[1], X.shape[2])
    model = model_class(input_shape)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            x, y, z = batch
            x = x.to(device)
            z = z.to(device).unsqueeze(2)
            
            optimizer.zero_grad()
            output = model(x)

            # Select loss function based on config
            if loss_type == "circular_mse":
                loss = circular_mse_loss(output, z)
            elif loss_type == "cosine_similarity":
                loss = cosine_similarity_loss(output, z)
            else:  # combined_loss
                loss = combined_loss(output, z)
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training completed!")
    return model


def main():
    # Load configuration
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load data
    X, Y, Z = load_data_percentage('./data/X.npy', './data/Y.npy', './data/Z.npy', percentage=100)
    dataset = CustomDataset(X, Y, Z)

    # Train the model
    trained_model = train_model(dataset, AttentionResNet56, config)

    # Save the trained model
    torch.save(trained_model.state_dict(), f"{config.model_path}{config.loss_type}.pth")

main()