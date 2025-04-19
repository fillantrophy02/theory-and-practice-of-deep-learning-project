import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

def train_model(model, train_dataloader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on {device}...")
    model.train()
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Use tqdm for progress display but only in the console
        progress_bar = tqdm(train_dataloader, 
                           desc=f"Epoch {epoch+1}/{num_epochs}",
                           leave=False)  # Don't leave progress bars
        
        for inputs, targets in progress_bar:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Fixing the TARGET and OUTPUT Sizing issue
            targets = targets.squeeze(1) 
            
            # Forward pass
            _, _, outputs = model(inputs)
            loss = criterion(outputs.float(), targets.float())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.numel()
        
        # Only print the summary for each epoch
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = 100 * correct_predictions / total_predictions
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s")
    
    return model