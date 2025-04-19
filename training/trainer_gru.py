import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd 
from config_custom.config_gru import CONFIG
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model_dataloader(model, train_dataloader, num_epochs=CONFIG['num_epochs'], learning_rate=CONFIG['learning_rate']):
    losses = []
    accuracies = []

    # Step 1: Compute positive class weight (No Rain / Rain)
    labels = torch.cat([Y for _, Y in train_dataloader])  #  Use training labels
    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()
    weight_ratio = num_neg / num_pos

    # Step 2: Weighted loss
    pos_weight = torch.tensor([weight_ratio]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_loss = float("inf")
    os.makedirs("ckpts/gru", exist_ok=True)
    best_model_path = "ckpts/gru/model_weights_gru.pth"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, Y_batch in train_dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()

            _, outputs = model(X_batch)
            outputs = outputs.squeeze(-1)
            predictions = outputs[:, -1].unsqueeze(1)

            loss = criterion(predictions, Y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            probs = torch.sigmoid(predictions.squeeze())
            predicted_labels = (probs >= 0.5).float()

            correct += (predicted_labels == Y_batch.squeeze(1)).sum().item()
            total += Y_batch.size(0)

        epoch_loss = total_loss / len(train_dataloader)
        epoch_accuracy = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)

    return losses, accuracies
