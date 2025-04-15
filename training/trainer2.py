import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd 

from config.config import CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model2(model, X_train, Y_train, num_epochs=CONFIG['num_epochs'], batch_size=CONFIG['batch_size'], learning_rate=CONFIG['learning_rate']):
    # Defining the Loss Function + Optimizer used
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create Mini Batch Data
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad()

            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # Forward pass 
            _, _, predictions = model(X_batch)
            
            loss = criterion(predictions.squeeze(-1), Y_batch.squeeze(-1))
            total_loss += loss.item()
            
            probs = torch.sigmoid(predictions.squeeze(-1))
            predicted_labels = (probs >= 0.5).float()
            
            correct += (predicted_labels == Y_batch.squeeze(-1)).sum().item()
            total += Y_batch.numel()  # Total number of elements in Y_batch

            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        torch.save(model.state_dict(), 'model2_weights.pth')
    
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Over Epochs")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()