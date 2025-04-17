import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd 

from config.config import CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, X_train, Y_train, num_epochs=CONFIG['num_epochs'], batch_size=CONFIG['batch_size'], learning_rate=CONFIG['learning_rate']):
    losses = []
    accuracies = []

    # Defining the Loss Function + Optimizer used
    ## Loss --> BCE
    ## Optimizer --> Adam

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create Mini Batch Data.
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):

        model.train() 
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, Y_batch in dataloader:

            optimizer.zero_grad()

            # Move to GPU if have.
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            

            # Forward pass
            hidden = torch.zeros(X_batch.size(0), model.hidden_size).to(device)
            _, predictions = model(X_batch, hidden)
            
            # Compute loss
            loss = criterion(predictions, Y_batch)
            total_loss += loss.item()
            
            # Accuracy Calculation
            probs = torch.sigmoid(predictions.squeeze())

            # OWN DEFINITION: if probability > 0.5 , gives a value of 1. This means it will RAIN.
            predicted_labels = (probs >= 0.5).float()

            correct += (predicted_labels == Y_batch).sum().item()
            total += Y_batch.size(0)

            # Backpropagation
            loss.backward()
            optimizer.step()
        
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        
        torch.save(model.state_dict(), 'model_weights.pth')
    return losses, accuracies
